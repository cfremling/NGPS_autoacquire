// ngps_acq.c
// Portable (macOS + Linux) acquisition helper for NGPS slice viewing camera.
// Uses directory polling OR single-file (one-shot) mode.
//
// Build (macOS Homebrew typical):
//   gcc -O3 -Wall -Wextra -o ngps_acq ngps_acq.c -lcfitsio -lwcs -lm
//
// Directory mode (polling):
//   ./ngps_acq --input /path/to/stream_dir --goal-x 512 --goal-y 512 --dry-run 1
//
// One-shot file mode:
//   ./ngps_acq --input /path/to/frame.fits --goal-x 512 --goal-y 512 --dry-run 1
//
// Multi-extension support:
// - Prefers EXTNAME="L" by default (left), falls back to extnum=1 if not found.

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <strings.h>   // strcasecmp
#include <math.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <dirent.h>
#include <sys/stat.h>
#include <limits.h>

#include "fitsio.h"
#include <wcslib/wcs.h>
#include <wcslib/wcshdr.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

typedef struct {
  char   input[PATH_MAX];  // directory OR file
  char   dir[PATH_MAX];    // if directory mode, this is the directory used
  int    oneshot;          // 1 => file mode; 0 => directory polling mode

  double goal_x;
  double goal_y;
  int    pixel_origin;     // 0 or 1
  double max_dist_pix;
  double tol_pix;
  double snr_thresh;
  int    min_adjacent;
  int    centroid_halfwin;
  int    max_iters;

  // FITS selection
  int    extnum;           // fallback: 0=primary, 1=first extension, etc.
  char   extname[32];      // preferred: e.g. "L" (left). Empty => use extnum.

  double poll_sec;         // directory scan interval
  int    dry_run;
  int    verbose;
} AcqParams;

typedef struct {
  int    found;
  double peak_x;    // pixel coords in same origin as params (0 or 1)
  double peak_y;
  double cx;        // centroid x
  double cy;        // centroid y
  double snr;
  double peak;
  double bkg;
  double sigma;
} Detection;

static void die(const char* msg) {
  fprintf(stderr, "FATAL: %s\n", msg);
  exit(4);
}

static int ends_with(const char* s, const char* suf) {
  size_t ns = strlen(s), nf = strlen(suf);
  if (nf > ns) return 0;
  return (strncmp(s + (ns - nf), suf, nf) == 0);
}

static int is_fits_name(const char* name) {
  return ends_with(name, ".fits") || ends_with(name, ".fit") || ends_with(name, ".fts");
}

static int cmp_float(const void* a, const void* b) {
  float fa = *(const float*)a;
  float fb = *(const float*)b;
  return (fa < fb) ? -1 : (fa > fb) ? 1 : 0;
}

static void sleep_seconds(double sec) {
  if (sec <= 0) return;
  struct timespec ts;
  ts.tv_sec  = (time_t)floor(sec);
  ts.tv_nsec = (long)((sec - (double)ts.tv_sec) * 1e9);
  while (nanosleep(&ts, &ts) == -1 && errno == EINTR) {}
}

static int path_is_dir(const char* p) {
  struct stat st;
  if (stat(p, &st) != 0) return 0;
  return S_ISDIR(st.st_mode);
}

static int path_is_file(const char* p) {
  struct stat st;
  if (stat(p, &st) != 0) return 0;
  return S_ISREG(st.st_mode);
}

// robust median + MAD using subsample stride
static void robust_median_mad(const float* img, long nx, long ny,
                             double* med_out, double* sigma_out)
{
  long N = nx * ny;
  long target = 200000;
  long stride = (N > target) ? (N / target) : 1;
  if (stride < 1) stride = 1;

  long ns = (N + stride - 1) / stride;
  float* sample = (float*)malloc((size_t)ns * sizeof(float));
  if (!sample) die("malloc sample failed");

  long k = 0;
  for (long i = 0; i < N; i += stride) sample[k++] = img[i];
  ns = k;

  qsort(sample, (size_t)ns, sizeof(float), cmp_float);
  double med = (ns % 2) ? sample[ns/2] : 0.5*(sample[ns/2 - 1] + sample[ns/2]);

  for (long i = 0; i < ns; i++) sample[i] = (float)fabs((double)sample[i] - med);
  qsort(sample, (size_t)ns, sizeof(float), cmp_float);
  double mad = (ns % 2) ? sample[ns/2] : 0.5*(sample[ns/2 - 1] + sample[ns/2]);

  free(sample);

  double sigma = 1.4826 * mad;
  if (!isfinite(sigma) || sigma <= 0) sigma = 1.0;

  *med_out = med;
  *sigma_out = sigma;
}

// Move to IMAGE HDU by EXTNAME match. Returns 0 on success, nonzero on failure.
static int move_to_image_hdu_by_extname(fitsfile* fptr, const char* want_extname, int* out_hdu_index, int* status)
{
  if (!want_extname || want_extname[0] == '\0') return 1;

  int nhdus = 0;
  if (fits_get_num_hdus(fptr, &nhdus, status)) return 2;

  for (int hdu = 1; hdu <= nhdus; hdu++) {
    int hdutype = 0;
    if (fits_movabs_hdu(fptr, hdu, &hdutype, status)) return 3;
    if (hdutype != IMAGE_HDU) continue;

    char extname[FLEN_VALUE] = {0};
    int keystat = 0;
    if (fits_read_key(fptr, TSTRING, "EXTNAME", extname, NULL, &keystat)) {
      extname[0] = '\0';
    }

    if (extname[0] && strcasecmp(extname, want_extname) == 0) {
      if (out_hdu_index) *out_hdu_index = hdu;
      return 0;
    }
  }

  return 4; // not found
}

// Read 2D float image + header string from preferred EXTNAME (if set), else extnum.
static int read_fits_image_and_header(const char* path, const AcqParams* p,
                                      float** img_out, long* nx_out, long* ny_out,
                                      char** header_out, int* nkeys_out,
                                      int* used_hdu_out, char* used_extname_out, size_t used_extname_sz)
{
  fitsfile* fptr = NULL;
  int status = 0;

  if (fits_open_file(&fptr, path, READONLY, &status)) return status;

  int used_hdu = 1;
  char used_extname[FLEN_VALUE] = {0};

  // Prefer EXTNAME selection if provided
  if (p->extname[0]) {
    int found_hdu = 0;
    int rc = move_to_image_hdu_by_extname(fptr, p->extname, &found_hdu, &status);
    if (rc == 0) {
      used_hdu = found_hdu;
    } else {
      if (p->verbose) fprintf(stderr, "WARNING: EXTNAME='%s' not found; falling back to extnum=%d\n", p->extname, p->extnum);
      status = 0;
      int hdutype = 0;
      if (fits_movabs_hdu(fptr, p->extnum + 1, &hdutype, &status)) {
        fits_close_file(fptr, &status);
        return status;
      }
      used_hdu = p->extnum + 1;
    }
  } else {
    int hdutype = 0;
    if (fits_movabs_hdu(fptr, p->extnum + 1, &hdutype, &status)) {
      fits_close_file(fptr, &status);
      return status;
    }
    used_hdu = p->extnum + 1;
  }

  // Record actual EXTNAME (if present)
  {
    int keystat = 0;
    if (fits_read_key(fptr, TSTRING, "EXTNAME", used_extname, NULL, &keystat)) {
      used_extname[0] = '\0';
    }
  }

  int bitpix = 0, naxis = 0;
  long naxes[3] = {0,0,0};
  if (fits_get_img_param(fptr, 3, &bitpix, &naxis, naxes, &status)) {
    fits_close_file(fptr, &status);
    return status;
  }
  if (naxis < 2) {
    fits_close_file(fptr, &status);
    return BAD_NAXIS;
  }

  long nx = naxes[0];
  long ny = naxes[1];

  float* img = (float*)malloc((size_t)nx * (size_t)ny * sizeof(float));
  if (!img) {
    fits_close_file(fptr, &status);
    return MEMORY_ALLOCATION;
  }

  long fpixel[3] = {1,1,1};
  long nelem = nx * ny;
  int anynul = 0;
  if (fits_read_pix(fptr, TFLOAT, fpixel, nelem, NULL, img, &anynul, &status)) {
    free(img);
    fits_close_file(fptr, &status);
    return status;
  }

  char* header = NULL;
  int nkeys = 0;
  int hstatus = 0;
  if (fits_hdr2str(fptr, 1, NULL, 0, &header, &nkeys, &hstatus)) {
    header = NULL;
    nkeys = 0;
  }

  fits_close_file(fptr, &status);

  *img_out = img;
  *nx_out = nx;
  *ny_out = ny;
  *header_out = header;
  *nkeys_out = nkeys;

  if (used_hdu_out) *used_hdu_out = used_hdu;
  if (used_extname_out && used_extname_sz > 0) snprintf(used_extname_out, used_extname_sz, "%s", used_extname);
  return 0;
}

// Init WCS from header string. Caller must wcsvfree(&nwcs, &wcs).
static int init_wcs_from_header(const char* header, int nkeys,
                                struct wcsprm** wcs_out, int* nwcs_out)
{
  *wcs_out = NULL;
  *nwcs_out = 0;

  if (!header || nkeys <= 0) return 1;

  int relax = WCSHDR_all;
  int ctrl  = 2;
  int nrej = 0, nwcs = 0;
  struct wcsprm* wcs = NULL;

  int stat = wcspih((char*)header, nkeys, relax, ctrl, &nrej, &nwcs, &wcs);
  if (stat || nwcs < 1 || !wcs) return 2;

  if (wcsset(&wcs[0])) {
    wcsvfree(&nwcs, &wcs);
    return 3;
  }

  *wcs_out = wcs;
  *nwcs_out = nwcs;
  return 0;
}

// Pixel -> (RA,Dec) degrees using WCSLIB. Inputs are FITS 1-based pixels.
static int pix2world_wcs(const struct wcsprm* wcs0, double pix_x, double pix_y,
                         double* ra_deg, double* dec_deg)
{
  if (!wcs0) return 1;

  double pixcrd[2] = {pix_x, pix_y};
  double imgcrd[2] = {0,0};
  double phi=0, theta=0;
  double world[2] = {0,0};
  int stat[2] = {0,0};

  int rc = wcsp2s((struct wcsprm*)wcs0, 1, 2, pixcrd, imgcrd, &phi, &theta, world, stat);
  if (rc) return 2;

  *ra_deg  = world[0];
  *dec_deg = world[1];
  return 0;
}

static double wrap_dra_deg(double dra) {
  while (dra > 180.0) dra -= 360.0;
  while (dra < -180.0) dra += 360.0;
  return dra;
}

// Replace with your real TCS interface if desired
static int tcs_move_arcsec(double dra_arcsec, double ddec_arcsec, int dry_run, int verbose)
{
  char cmd[512];
  snprintf(cmd, sizeof(cmd), "tcs move ra %.3f dec %.3f", dra_arcsec, ddec_arcsec);

  if (verbose || dry_run) fprintf(stderr, "TCS CMD: %s\n", cmd);
  if (dry_run) return 0;

  int rc = system(cmd);
  if (rc != 0) {
    fprintf(stderr, "WARNING: TCS command returned %d\n", rc);
    return 1;
  }
  return 0;
}

// Detect star near goal
static Detection detect_star_near_goal(const float* img, long nx, long ny, const AcqParams* p)
{
  Detection d;
  memset(&d, 0, sizeof(d));

  double med=0, sigma=0;
  robust_median_mad(img, nx, ny, &med, &sigma);

  const double thr = med + p->snr_thresh * sigma;

  const double goal_x0 = (p->pixel_origin == 0) ? p->goal_x : (p->goal_x - 1.0);
  const double goal_y0 = (p->pixel_origin == 0) ? p->goal_y : (p->goal_y - 1.0);

  long x0 = (long)floor(goal_x0 - p->max_dist_pix);
  long x1 = (long)ceil (goal_x0 + p->max_dist_pix);
  long y0 = (long)floor(goal_y0 - p->max_dist_pix);
  long y1 = (long)ceil (goal_y0 + p->max_dist_pix);

  if (x0 < 1) x0 = 1;
  if (y0 < 1) y0 = 1;
  if (x1 > nx-2) x1 = nx-2;
  if (y1 > ny-2) y1 = ny-2;

  double best_val = -1e300;
  long best_x = -1, best_y = -1;
  double best_snr = 0;

  for (long y = y0; y <= y1; y++) {
    for (long x = x0; x <= x1; x++) {
      double v = img[y*nx + x];
      if (v <= thr) continue;

      // local max in 3x3
      int ismax = 1;
      for (int dy = -1; dy <= 1 && ismax; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
          if (dx == 0 && dy == 0) continue;
          if (img[(y+dy)*nx + (x+dx)] >= v) { ismax = 0; break; }
        }
      }
      if (!ismax) continue;

      // adjacent above threshold
      int nadj = 0;
      for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
          if (dx == 0 && dy == 0) continue;
          if (img[(y+dy)*nx + (x+dx)] > thr) nadj++;
        }
      }
      if (nadj < p->min_adjacent) continue;

      double dxg = (double)x - goal_x0;
      double dyg = (double)y - goal_y0;
      double r = hypot(dxg, dyg);
      if (r > p->max_dist_pix) continue;

      double snr_here = (v - med) / sigma;

      if (v > best_val) {
        best_val = v;
        best_x = x;
        best_y = y;
        best_snr = snr_here;
      }
    }
  }

  d.bkg = med;
  d.sigma = sigma;

  if (best_x < 0) {
    d.found = 0;
    return d;
  }

  d.found = 1;
  d.peak = best_val;
  d.snr = best_snr;

  // centroid in window
  int hw = p->centroid_halfwin;
  long cx0 = best_x - hw; if (cx0 < 0) cx0 = 0;
  long cx1 = best_x + hw; if (cx1 > nx-1) cx1 = nx-1;
  long cy0 = best_y - hw; if (cy0 < 0) cy0 = 0;
  long cy1 = best_y + hw; if (cy1 > ny-1) cy1 = ny-1;

  double sumI = 0, sumX = 0, sumY = 0;
  for (long y = cy0; y <= cy1; y++) {
    for (long x = cx0; x <= cx1; x++) {
      double I = (double)img[y*nx + x] - med;
      if (I <= 0) continue;
      sumI += I;
      sumX += I * (double)x;
      sumY += I * (double)y;
    }
  }

  double cxo = (sumI > 0) ? (sumX / sumI) : (double)best_x;
  double cyo = (sumI > 0) ? (sumY / sumI) : (double)best_y;

  d.peak_x = (p->pixel_origin == 0) ? (double)best_x : (double)best_x + 1.0;
  d.peak_y = (p->pixel_origin == 0) ? (double)best_y : (double)best_y + 1.0;
  d.cx     = (p->pixel_origin == 0) ? cxo          : cxo + 1.0;
  d.cy     = (p->pixel_origin == 0) ? cyo          : cyo + 1.0;

  return d;
}

// --- Directory polling ---
typedef struct {
  time_t mtime;
  off_t  size;
  char   name[NAME_MAX+1];
} FileKey;

static int key_is_newer(const FileKey* a, const FileKey* b) {
  if (a->mtime != b->mtime) return (a->mtime > b->mtime);
  if (a->size  != b->size)  return (a->size  > b->size);
  return (strcmp(a->name, b->name) > 0);
}

static int find_newer_fits(const char* dir, const FileKey* last,
                           char* out_path, size_t out_sz, FileKey* out_key)
{
  DIR* dp = opendir(dir);
  if (!dp) return -1;

  struct dirent* de;
  int found = 0;
  FileKey best = *last;
  char best_path[PATH_MAX] = {0};

  while ((de = readdir(dp)) != NULL) {
    if (de->d_name[0] == '.') continue;
    if (!is_fits_name(de->d_name)) continue;

    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s/%s", dir, de->d_name);

    struct stat st;
    if (stat(path, &st) != 0) continue;
    if (st.st_size <= 0) continue;

    FileKey k;
    k.mtime = st.st_mtime;
    k.size  = st.st_size;
    snprintf(k.name, sizeof(k.name), "%s", de->d_name);

    if (key_is_newer(&k, &best)) {
      best = k;
      snprintf(best_path, sizeof(best_path), "%s", path);
      found = 1;
    }
  }

  closedir(dp);

  if (!found) return 1;

  snprintf(out_path, out_sz, "%s", best_path);
  *out_key = best;
  return 0;
}

static int wait_for_next_fits_poll(const char* dir, double poll_sec,
                                   char* out_path, size_t out_sz, int verbose)
{
  static FileKey last = {0, 0, {0}};

  for (;;) {
    FileKey k;
    int rc = find_newer_fits(dir, &last, out_path, out_sz, &k);
    if (rc < 0) return rc;

    if (rc == 0) {
      struct stat st1, st2;
      if (stat(out_path, &st1) == 0 && st1.st_size > 0) {
        sleep_seconds(0.15);
        if (stat(out_path, &st2) == 0 && st2.st_size == st1.st_size) {
          last = k;
          if (verbose) fprintf(stderr, "New FITS: %s\n", out_path);
          return 0;
        }
      }
    }

    sleep_seconds(poll_sec);
  }
}

// CLI parsing
static void usage(const char* argv0) {
  fprintf(stderr,
    "Usage: %s --input PATH --goal-x X --goal-y Y [options]\n"
    "PATH can be a directory (polling mode) or a FITS file (one-shot mode).\n"
    "Back-compat: --dir PATH works too.\n"
    "Options:\n"
    "  --pixel-origin 0|1      Pixel origin for goal and centroid (default 0)\n"
    "  --max-dist PIX          Search radius around goal (default 200)\n"
    "  --tol PIX               Success tolerance radius (default 0.5)\n"
    "  --snr S                 SNR threshold (default 8)\n"
    "  --min-adj N             Min adjacent pixels above threshold (default 4)\n"
    "  --centroid-hw N         Half-width centroid window (default 6)\n"
    "  --max-iters N           Max iterations (directory mode) (default 10)\n"
    "  --extname NAME          Preferred image EXTNAME (default L)\n"
    "  --extnum N              Fallback HDU (0=primary, 1=first ext...) (default 1)\n"
    "  --poll-sec S            Poll interval seconds (directory mode) (default 1.0)\n"
    "  --dry-run 0|1           Do not call TCS (default 0)\n"
    "  --verbose 0|1           Verbose logging (default 1)\n",
    argv0
  );
}

static void set_defaults(AcqParams* p) {
  memset(p, 0, sizeof(*p));
  snprintf(p->input, sizeof(p->input), ".");
  snprintf(p->dir, sizeof(p->dir), ".");
  p->oneshot = 0;

  p->goal_x = 0;
  p->goal_y = 0;
  p->pixel_origin = 0;
  p->max_dist_pix = 200.0;
  p->tol_pix = 0.5;
  p->snr_thresh = 8.0;
  p->min_adjacent = 4;
  p->centroid_halfwin = 6;
  p->max_iters = 10;

  snprintf(p->extname, sizeof(p->extname), "L");
  p->extnum = 1;

  p->poll_sec = 1.0;
  p->dry_run = 0;
  p->verbose = 1;
}

static int parse_args(int argc, char** argv, AcqParams* p) {
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--input") && i+1 < argc) {
      snprintf(p->input, sizeof(p->input), "%s", argv[++i]);
    } else if (!strcmp(argv[i], "--dir") && i+1 < argc) { // back-compat alias
      snprintf(p->input, sizeof(p->input), "%s", argv[++i]);
    } else if (!strcmp(argv[i], "--goal-x") && i+1 < argc) {
      p->goal_x = atof(argv[++i]);
    } else if (!strcmp(argv[i], "--goal-y") && i+1 < argc) {
      p->goal_y = atof(argv[++i]);
    } else if (!strcmp(argv[i], "--pixel-origin") && i+1 < argc) {
      p->pixel_origin = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "--max-dist") && i+1 < argc) {
      p->max_dist_pix = atof(argv[++i]);
    } else if (!strcmp(argv[i], "--tol") && i+1 < argc) {
      p->tol_pix = atof(argv[++i]);
    } else if (!strcmp(argv[i], "--snr") && i+1 < argc) {
      p->snr_thresh = atof(argv[++i]);
    } else if (!strcmp(argv[i], "--min-adj") && i+1 < argc) {
      p->min_adjacent = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "--centroid-hw") && i+1 < argc) {
      p->centroid_halfwin = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "--max-iters") && i+1 < argc) {
      p->max_iters = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "--extnum") && i+1 < argc) {
      p->extnum = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "--extname") && i+1 < argc) {
      snprintf(p->extname, sizeof(p->extname), "%s", argv[++i]);
      if (!strcasecmp(p->extname, "none")) p->extname[0] = '\0';
    } else if (!strcmp(argv[i], "--poll-sec") && i+1 < argc) {
      p->poll_sec = atof(argv[++i]);
    } else if (!strcmp(argv[i], "--dry-run") && i+1 < argc) {
      p->dry_run = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "--verbose") && i+1 < argc) {
      p->verbose = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "--help")) {
      return 0;
    } else {
      fprintf(stderr, "Unknown/invalid arg: %s\n", argv[i]);
      return -1;
    }
  }

  if (p->goal_x == 0 && p->goal_y == 0) {
    fprintf(stderr, "You must provide --goal-x and --goal-y\n");
    return -1;
  }
  if (!(p->pixel_origin == 0 || p->pixel_origin == 1)) {
    fprintf(stderr, "--pixel-origin must be 0 or 1\n");
    return -1;
  }
  if (p->poll_sec <= 0) p->poll_sec = 1.0;
  return 1;
}

// Process one FITS file once. Returns exit code (0/1/2/3/4).
static int process_one_file(const char* fits_path, const AcqParams* p)
{
  float* img = NULL;
  long nx=0, ny=0;
  char* header = NULL;
  int nkeys = 0;
  int used_hdu = 0;
  char used_extname[64] = {0};

  int st = read_fits_image_and_header(fits_path, p, &img, &nx, &ny, &header, &nkeys,
                                      &used_hdu, used_extname, sizeof(used_extname));
  if (st) {
    fprintf(stderr, "ERROR: CFITSIO read failed for %s (status=%d)\n", fits_path, st);
    if (img) free(img);
    if (header) free(header);
    return 4;
  }

  if (p->verbose) {
    fprintf(stderr, "Using FITS=%s HDU=%d EXTNAME=%s size=%ldx%ld\n",
            fits_path, used_hdu, (used_extname[0] ? used_extname : "(none)"), nx, ny);
  }

  Detection d = detect_star_near_goal(img, nx, ny, p);
  if (!d.found) {
    fprintf(stderr, "No suitable star detected near goal.\n");
    printf("NGPS_ACQ_RESULT found=0\n");
    free(img);
    if (header) free(header);
    return 2;
  }

  double dx = d.cx - p->goal_x;
  double dy = d.cy - p->goal_y;
  double r  = hypot(dx, dy);

  // WCS
  struct wcsprm* wcs = NULL;
  int nwcs = 0;
  int wcs_stat = init_wcs_from_header(header, nkeys, &wcs, &nwcs);
  if (wcs_stat != 0) {
    fprintf(stderr, "ERROR: WCS parse failed (code=%d).\n", wcs_stat);
    printf("NGPS_ACQ_RESULT found=1 cx=%.6f cy=%.6f dx_pix=%.6f dy_pix=%.6f r_pix=%.6f snr=%.3f wcs_ok=0\n",
           d.cx, d.cy, dx, dy, r, d.snr);
    free(img);
    if (header) free(header);
    return 3;
  }

  double goal_x1 = (p->pixel_origin == 0) ? (p->goal_x + 1.0) : p->goal_x;
  double goal_y1 = (p->pixel_origin == 0) ? (p->goal_y + 1.0) : p->goal_y;
  double star_x1 = (p->pixel_origin == 0) ? (d.cx + 1.0)     : d.cx;
  double star_y1 = (p->pixel_origin == 0) ? (d.cy + 1.0)     : d.cy;

  double ra_goal=0, dec_goal=0, ra_star=0, dec_star=0;
  if (pix2world_wcs(&wcs[0], goal_x1, goal_y1, &ra_goal, &dec_goal) ||
      pix2world_wcs(&wcs[0], star_x1, star_y1, &ra_star, &dec_star)) {
    fprintf(stderr, "ERROR: WCS pix2world failed.\n");
    wcsvfree(&nwcs, &wcs);
    free(img);
    if (header) free(header);
    return 3;
  }

  double dra_deg  = wrap_dra_deg(ra_goal - ra_star);
  double ddec_deg = (dec_goal - dec_star);

  double cosdec = cos(dec_goal * M_PI / 180.0);
  double dra_arcsec  = dra_deg  * 3600.0 * cosdec;
  double ddec_arcsec = ddec_deg * 3600.0;

  // machine-readable line for integration
  printf("NGPS_ACQ_RESULT found=1 cx=%.6f cy=%.6f dx_pix=%.6f dy_pix=%.6f r_pix=%.6f dra_arcsec=%.6f ddec_arcsec=%.6f snr=%.3f wcs_ok=1\n",
         d.cx, d.cy, dx, dy, r, dra_arcsec, ddec_arcsec, d.snr);

  if (p->verbose) {
    fprintf(stderr,
      "Star found: centroid=(%.3f,%.3f) dx=%.3f dy=%.3f r=%.3f pix SNR=%.2f\n",
      d.cx, d.cy, dx, dy, r, d.snr
    );
    fprintf(stderr, "Command offsets: dra=%.3f\" ddec=%.3f\"\n", dra_arcsec, ddec_arcsec);
  }

  if (r <= p->tol_pix) {
    if (p->verbose) fprintf(stderr, "SUCCESS: within tolerance (%.2f pix)\n", p->tol_pix);
    wcsvfree(&nwcs, &wcs);
    free(img);
    if (header) free(header);
    return 0;
  }

  (void)tcs_move_arcsec(dra_arcsec, ddec_arcsec, p->dry_run, p->verbose);

  wcsvfree(&nwcs, &wcs);
  free(img);
  if (header) free(header);
  return 1;
}

int main(int argc, char** argv)
{
  AcqParams p;
  set_defaults(&p);

  int pr = parse_args(argc, argv, &p);
  if (pr <= 0) { usage(argv[0]); return (pr == 0) ? 0 : 4; }

  if (path_is_file(p.input)) {
    p.oneshot = 1;
  } else if (path_is_dir(p.input)) {
    p.oneshot = 0;
    snprintf(p.dir, sizeof(p.dir), "%s", p.input);
  } else {
    fprintf(stderr, "ERROR: --input path does not exist or is not a file/dir: %s\n", p.input);
    return 4;
  }

  if (p.verbose) {
    fprintf(stderr,
      "NGPS Acquisition starting:\n"
      "  input=%s mode=%s goal=(%.3f,%.3f) origin=%d max_dist=%.1f tol=%.2f snr=%.1f min_adj=%d hw=%d extname=%s extnum=%d poll=%.2fs\n",
      p.input, (p.oneshot ? "oneshot-file" : "dir-poll"),
      p.goal_x, p.goal_y, p.pixel_origin, p.max_dist_pix, p.tol_pix,
      p.snr_thresh, p.min_adjacent, p.centroid_halfwin,
      (p.extname[0] ? p.extname : "(none)"), p.extnum, p.poll_sec
    );
  }

  if (p.oneshot) {
    // process a single file once and exit
    return process_one_file(p.input, &p);
  }

  // directory polling mode: iterate for max_iters frames
  for (int iter = 1; iter <= p.max_iters; iter++) {
    char path[PATH_MAX];

    // wait for next FITS in directory
    // (reuse previous polling helper)
    // NOTE: simplest: inline call to wait_for_next_fits_poll
    // (keeping it local here for clarity).
    extern int wait_for_next_fits_poll(const char*, double, char*, size_t, int);

    int wrc = wait_for_next_fits_poll(p.dir, p.poll_sec, path, sizeof(path), p.verbose);
    if (wrc != 0) die("Failed waiting for next FITS (polling).");

    if (p.verbose) fprintf(stderr, "Iter %d processing %s\n", iter, path);

    int rc = process_one_file(path, &p);
    if (rc == 0) return 0;   // reached tolerance
    // else keep looping in directory mode regardless of rc (no-star/WCS failure will just try next frame)
  }

  fprintf(stderr, "FAILED: reached max iterations (%d) in directory mode.\n", p.max_iters);
  return 1;
}
