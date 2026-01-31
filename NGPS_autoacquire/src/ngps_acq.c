// ngps_acq.c
// NGPS acquisition helper for the slice-viewing camera.
//
// Build:
//   gcc -O3 -Wall -Wextra -std=c11 -o ngps_acq ngps_acq.c -lcfitsio -lwcs -lm
//
// One-shot (compute offsets, optionally move once):
//   ./ngps_acq --input /path/to/frame.fits --goal-x 512 --goal-y 512 --dry-run 1
//
// Closed-loop acquisition:
//   ./ngps_acq --input /path/to/live.fits --goal-x 512 --goal-y 512 --loop 1
//
// Notes:
//  - This program assumes the camera writes/overwrites the same FITS file.
//  - Star detection and centroiding are designed to be robust under poor seeing.
//  - Offsets returned are in arcsec and intended for:  tcs native pt <dra> <ddec>
//
// Diagnostics:
//  - Use --debug 1 to write a PPM overlay of the background ROI:
//      * thresholded pixels colored
//      * peak circle, centroid +, goal x, arrow to goal
//
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <signal.h>
#include <sys/stat.h>
#include <limits.h>

#include "fitsio.h"
#include <wcslib/wcs.h>
#include <wcslib/wcshdr.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

// ROI mask bits
#define ROI_X1_SET 0x01
#define ROI_X2_SET 0x02
#define ROI_Y1_SET 0x04
#define ROI_Y2_SET 0x08

typedef struct {
  char   input[PATH_MAX];

  double goal_x;
  double goal_y;
  int    pixel_origin;        // 0 or 1

  // search constraints
  double max_dist_pix;        // circular cut around goal
  double snr_thresh;          // detection threshold in sigma
  int    min_adjacent;        // raw-pixel neighbors above threshold

  // detection filter
  double filt_sigma_pix;      // Gaussian sigma for smoothing (pix)

  // centroiding
  int    centroid_halfwin;    // window half-width (pix)
  double centroid_sigma_pix;  // Gaussian window sigma (pix)
  int    centroid_maxiter;
  double centroid_eps_pix;

  // FITS selection
  int    extnum;              // fallback: 0=primary, 1=first extension
  char   extname[32];         // preferred EXTNAME ("L" default). "none" disables.

  // Background statistics ROI (typically the illuminated / on-sky region)
  int    bg_roi_mask;
  long   bg_x1, bg_x2;
  long   bg_y1, bg_y2;

  // Candidate search ROI (defaults to bg ROI if unset)
  int    search_roi_mask;
  long   search_x1, search_x2;
  long   search_y1, search_y2;

  // Closed-loop wrapper
  int    loop;
  double cadence_sec;         // seconds between accepted samples
  int    max_samples;         // per-move gather
  int    min_samples;         // minimum before testing precision
  double prec_arcsec;         // required scatter (MAD->sigma) per axis
  double goal_arcsec;         // convergence threshold on robust offset magnitude
  int    max_cycles;          // max move cycles
  double gain;                // multiply commanded move (0..1 recommended)

  // WCS/offset conventions
  int    dra_use_cosdec;      // 1: dra = dRA*cos(dec) (default); 0: dra = dRA
  int    tcs_sign;            // multiply commanded offsets by +/-1

  // TCS options
  int    tcs_set_units;       // if 1: run "tcs native dra 'arcsec'" and "... ddec 'arcsec'" once

  // Debug
  int    debug;
  char   debug_out[PATH_MAX];

  int    dry_run;
  int    verbose;
} AcqParams;

typedef struct {
  int    found;
  // peak in pixel coords (user origin)
  double peak_x;
  double peak_y;
  // windowed centroid (user origin)
  double cx;
  double cy;

  double peak_val;
  double peak_snr_raw;
  double snr_ap;      // aperture-like SNR within centroid window

  double bkg;
  double sigma;

  // For debug
  long   cand_x0, cand_y0; // 0-based
  long   cand_x1, cand_y1;
} Detection;

typedef struct {
  int    ok;
  Detection det;

  // pixel offsets
  double dx_pix;
  double dy_pix;

  // WCS
  int    wcs_ok;
  double ra_goal_deg, dec_goal_deg;
  double ra_star_deg, dec_star_deg;

  // Commanded offsets (arcsec) for tcs native pt <dra> <ddec>
  double dra_cmd_arcsec;
  double ddec_cmd_arcsec;
  double r_cmd_arcsec;
} FrameResult;

static volatile sig_atomic_t g_stop = 0;
static void on_sigint(int sig) { (void)sig; g_stop = 1; }

static void die(const char* msg) {
  fprintf(stderr, "FATAL: %s\n", msg);
  exit(4);
}

static void sleep_seconds(double sec) {
  if (sec <= 0) return;
  struct timespec ts;
  ts.tv_sec  = (time_t)floor(sec);
  ts.tv_nsec = (long)((sec - (double)ts.tv_sec) * 1e9);
  while (nanosleep(&ts, &ts) == -1 && errno == EINTR) {}
}

static int cmp_float(const void* a, const void* b) {
  float fa = *(const float*)a;
  float fb = *(const float*)b;
  return (fa < fb) ? -1 : (fa > fb) ? 1 : 0;
}

static int cmp_double(const void* a, const void* b) {
  double da = *(const double*)a;
  double db = *(const double*)b;
  return (da < db) ? -1 : (da > db) ? 1 : 0;
}

static double wrap_dra_deg(double dra) {
  while (dra > 180.0) dra -= 360.0;
  while (dra < -180.0) dra += 360.0;
  return dra;
}

// Convert a user-specified ROI (mask + bounds in user origin) into clamped 0-based inclusive bounds.
// If mask is 0, returns full-frame bounds.
static void compute_roi_0based(long nx, long ny, int pixel_origin,
                               int mask, long ux1_in, long ux2_in, long uy1_in, long uy2_in,
                               long* x1, long* x2, long* y1, long* y2)
{
  long ux1 = (pixel_origin == 0) ? 0  : 1;
  long ux2 = (pixel_origin == 0) ? (nx - 1) : nx;
  long uy1 = (pixel_origin == 0) ? 0  : 1;
  long uy2 = (pixel_origin == 0) ? (ny - 1) : ny;

  if (mask & ROI_X1_SET) ux1 = ux1_in;
  if (mask & ROI_X2_SET) ux2 = ux2_in;
  if (mask & ROI_Y1_SET) uy1 = uy1_in;
  if (mask & ROI_Y2_SET) uy2 = uy2_in;

  long ax1 = ux1, ax2 = ux2, ay1 = uy1, ay2 = uy2;
  if (pixel_origin == 1) { ax1--; ax2--; ay1--; ay2--; }

  if (ax2 < ax1) { long t=ax1; ax1=ax2; ax2=t; }
  if (ay2 < ay1) { long t=ay1; ay1=ay2; ay2=t; }

  if (ax1 < 0) ax1 = 0;
  if (ay1 < 0) ay1 = 0;
  if (ax2 > nx-1) ax2 = nx-1;
  if (ay2 > ny-1) ay2 = ny-1;

  if (ax2 < ax1) { ax1=0; ax2=nx-1; }
  if (ay2 < ay1) { ay1=0; ay2=ny-1; }

  *x1=ax1; *x2=ax2; *y1=ay1; *y2=ay2;
}

// Subsample pixels in ROI into a float array. Returns allocated array and count.
static float* roi_subsample(const float* img, long nx, long ny,
                            long x1, long x2, long y1, long y2,
                            long target, long* n_out)
{
  if (x1 < 0) x1 = 0;
  if (y1 < 0) y1 = 0;
  if (x2 > nx-1) x2 = nx-1;
  if (y2 > ny-1) y2 = ny-1;

  long wx = x2 - x1 + 1;
  long wy = y2 - y1 + 1;
  if (wx <= 0 || wy <= 0) { *n_out = 0; return NULL; }

  long Nroi = wx * wy;
  long stride = (Nroi > target) ? (Nroi / target) : 1;
  if (stride < 1) stride = 1;

  long ns = (Nroi + stride - 1) / stride;
  float* sample = (float*)malloc((size_t)ns * sizeof(float));
  if (!sample) die("malloc sample failed");

  long k = 0;
  long idx = 0;
  for (long y = y1; y <= y2; y++) {
    long row0 = y * nx;
    for (long x = x1; x <= x2; x++, idx++) {
      if ((idx % stride) == 0) sample[k++] = img[row0 + x];
    }
  }

  *n_out = k;
  return sample;
}

// SExtractor-like global background + sigma estimation within ROI:
//  - initial median + MAD
//  - iterative sigma-clipping around median
//  - background estimate via mode = 2.5*median - 1.5*mean, unless (mean-median)/sigma > 0.3 => use median
static void bg_sigma_sextractor_like(const float* img, long nx, long ny,
                                     long x1, long x2, long y1, long y2,
                                     double* bkg_out, double* sigma_out)
{
  *bkg_out = 0.0;
  *sigma_out = 1.0;

  long ns = 0;
  float* sample = roi_subsample(img, nx, ny, x1, x2, y1, y2, 200000, &ns);
  if (!sample || ns < 64) {
    if (sample) free(sample);
    return;
  }

  qsort(sample, (size_t)ns, sizeof(float), cmp_float);
  double median = (ns % 2) ? sample[ns/2] : 0.5*(sample[ns/2 - 1] + sample[ns/2]);

  // MAD
  float* dev = (float*)malloc((size_t)ns * sizeof(float));
  if (!dev) die("malloc dev failed");
  for (long i = 0; i < ns; i++) dev[i] = (float)fabs((double)sample[i] - median);
  qsort(dev, (size_t)ns, sizeof(float), cmp_float);
  double mad = (ns % 2) ? dev[ns/2] : 0.5*(dev[ns/2 - 1] + dev[ns/2]);
  free(dev);

  double sigma = 1.4826 * mad;
  if (!isfinite(sigma) || sigma <= 0) sigma = 1.0;

  // Iterative clip around median
  const double clip = 3.0;
  double mean = median;
  double sigma_prev = sigma;
  for (int it = 0; it < 8; it++) {
    double lo = median - clip * sigma;
    double hi = median + clip * sigma;

    double sum = 0.0, sum2 = 0.0;
    long n = 0;
    for (long i = 0; i < ns; i++) {
      double v = sample[i];
      if (v < lo || v > hi) continue;
      sum += v;
      sum2 += v*v;
      n++;
    }
    if (n < 32) break;

    mean = sum / (double)n;
    double var = (sum2 / (double)n) - mean*mean;
    if (var < 0) var = 0;
    sigma = sqrt(var);
    if (!isfinite(sigma) || sigma <= 0) sigma = sigma_prev;

    double rel = fabs(sigma - sigma_prev) / (sigma_prev > 0 ? sigma_prev : 1.0);
    sigma_prev = sigma;
    if (rel < 0.01) break;
  }

  // Mode estimator; fallback to median if strongly skewed
  double mode = 2.5*median - 1.5*mean;
  double bkg = mode;
  if (sigma > 0 && (mean - median)/sigma > 0.3) bkg = median;
  if (!isfinite(bkg)) bkg = median;

  if (!isfinite(sigma) || sigma <= 0) sigma = 1.0;

  *bkg_out = bkg;
  *sigma_out = sigma;

  free(sample);
}

static double median_of_doubles(double* a, int n)
{
  if (n <= 0) return 0.0;
  qsort(a, (size_t)n, sizeof(double), cmp_double);
  return (n % 2) ? a[n/2] : 0.5*(a[n/2 - 1] + a[n/2]);
}

static double mad_sigma_of_doubles(const double* a_in, int n, double med)
{
  if (n <= 1) return 0.0;
  double* d = (double*)malloc((size_t)n * sizeof(double));
  if (!d) die("malloc mad failed");
  for (int i = 0; i < n; i++) d[i] = fabs(a_in[i] - med);
  qsort(d, (size_t)n, sizeof(double), cmp_double);
  double mad = (n % 2) ? d[n/2] : 0.5*(d[n/2 - 1] + d[n/2]);
  free(d);
  return 1.4826 * mad;
}

// Gaussian kernel (1D) normalized to sum=1. Returns pointer and radius.
static double* make_gaussian_kernel(double sigma, int* radius_out)
{
  if (sigma <= 0.2) sigma = 0.2;
  int r = (int)ceil(3.0*sigma);
  if (r < 1) r = 1;
  int len = 2*r + 1;
  double* k = (double*)malloc((size_t)len * sizeof(double));
  if (!k) die("malloc kernel failed");

  double sum = 0.0;
  for (int i = -r; i <= r; i++) {
    double x = (double)i / sigma;
    double v = exp(-0.5 * x * x);
    k[i+r] = v;
    sum += v;
  }
  if (sum <= 0) sum = 1.0;
  for (int i = 0; i < len; i++) k[i] /= sum;

  *radius_out = r;
  return k;
}

static double kernel_sum_sq(const double* k, int radius)
{
  int len = 2*radius + 1;
  double s2 = 0.0;
  for (int i = 0; i < len; i++) s2 += k[i]*k[i];
  return s2;
}

// Separable convolution on a patch (width w, height h) with 1D kernel k (radius r).
// Input and output are float arrays length w*h. Border handling: clamp.
static void convolve_separable(const float* in, float* tmp, float* out, int w, int h, const double* k, int r)
{
  // horizontal into tmp
  for (int y = 0; y < h; y++) {
    const float* row = in + y*w;
    float* trow = tmp + y*w;
    for (int x = 0; x < w; x++) {
      double acc = 0.0;
      for (int dx = -r; dx <= r; dx++) {
        int xx = x + dx;
        if (xx < 0) xx = 0;
        if (xx >= w) xx = w-1;
        acc += (double)row[xx] * k[dx+r];
      }
      trow[x] = (float)acc;
    }
  }

  // vertical into out
  for (int y = 0; y < h; y++) {
    float* orow = out + y*w;
    for (int x = 0; x < w; x++) {
      double acc = 0.0;
      for (int dy = -r; dy <= r; dy++) {
        int yy = y + dy;
        if (yy < 0) yy = 0;
        if (yy >= h) yy = h-1;
        acc += (double)tmp[yy*w + x] * k[dy+r];
      }
      orow[x] = (float)acc;
    }
  }
}

// Draw helpers for debug PPM
static void set_px(uint8_t* rgb, int w, int h, int x, int y, uint8_t r, uint8_t g, uint8_t b)
{
  if (x < 0 || y < 0 || x >= w || y >= h) return;
  size_t idx = (size_t)(y*w + x) * 3;
  rgb[idx+0] = r;
  rgb[idx+1] = g;
  rgb[idx+2] = b;
}

static void draw_plus(uint8_t* rgb, int w, int h, int x, int y, int rad, uint8_t r, uint8_t g, uint8_t b)
{
  for (int dx = -rad; dx <= rad; dx++) set_px(rgb, w, h, x+dx, y, r,g,b);
  for (int dy = -rad; dy <= rad; dy++) set_px(rgb, w, h, x, y+dy, r,g,b);
}

static void draw_x(uint8_t* rgb, int w, int h, int x, int y, int rad, uint8_t r, uint8_t g, uint8_t b)
{
  for (int d = -rad; d <= rad; d++) {
    set_px(rgb, w, h, x+d, y+d, r,g,b);
    set_px(rgb, w, h, x+d, y-d, r,g,b);
  }
}

static void draw_circle(uint8_t* rgb, int w, int h, int xc, int yc, int rad, uint8_t r, uint8_t g, uint8_t b)
{
  // simple midpoint-ish sampling
  int x = rad;
  int y = 0;
  int err = 0;
  while (x >= y) {
    set_px(rgb,w,h, xc + x, yc + y, r,g,b);
    set_px(rgb,w,h, xc + y, yc + x, r,g,b);
    set_px(rgb,w,h, xc - y, yc + x, r,g,b);
    set_px(rgb,w,h, xc - x, yc + y, r,g,b);
    set_px(rgb,w,h, xc - x, yc - y, r,g,b);
    set_px(rgb,w,h, xc - y, yc - x, r,g,b);
    set_px(rgb,w,h, xc + y, yc - x, r,g,b);
    set_px(rgb,w,h, xc + x, yc - y, r,g,b);
    y++;
    if (err <= 0) {
      err += 2*y + 1;
    } else {
      x--;
      err += 2*(y - x) + 1;
    }
  }
}

static void draw_line(uint8_t* rgb, int w, int h, int x0, int y0, int x1, int y1, uint8_t r, uint8_t g, uint8_t b)
{
  int dx = abs(x1 - x0), sx = (x0 < x1) ? 1 : -1;
  int dy = -abs(y1 - y0), sy = (y0 < y1) ? 1 : -1;
  int err = dx + dy;
  while (1) {
    set_px(rgb,w,h,x0,y0,r,g,b);
    if (x0 == x1 && y0 == y1) break;
    int e2 = 2*err;
    if (e2 >= dy) { err += dy; x0 += sx; }
    if (e2 <= dx) { err += dx; y0 += sy; }
  }
}

static void draw_arrow(uint8_t* rgb, int w, int h, int x0, int y0, int x1, int y1, uint8_t r, uint8_t g, uint8_t b)
{
  draw_line(rgb,w,h,x0,y0,x1,y1,r,g,b);
  // arrowhead
  double ang = atan2((double)(y1 - y0), (double)(x1 - x0));
  double a1 = ang + 3.141592653589793/8.0;
  double a2 = ang - 3.141592653589793/8.0;
  int L = 10;
  int hx1 = x1 - (int)lround(L * cos(a1));
  int hy1 = y1 - (int)lround(L * sin(a1));
  int hx2 = x1 - (int)lround(L * cos(a2));
  int hy2 = y1 - (int)lround(L * sin(a2));
  draw_line(rgb,w,h,x1,y1,hx1,hy1,r,g,b);
  draw_line(rgb,w,h,x1,y1,hx2,hy2,r,g,b);
}

static int write_debug_ppm(const char* outpath,
                           const float* img, long nx, long ny,
                           long rx1, long rx2, long ry1, long ry2,
                           double bkg, double sigma, double snr_thresh,
                           const Detection* det,
                           const AcqParams* p)
{
  int w = (int)(rx2 - rx1 + 1);
  int h = (int)(ry2 - ry1 + 1);
  if (w <= 0 || h <= 0) return 1;

  uint8_t* rgb = (uint8_t*)malloc((size_t)w * (size_t)h * 3);
  if (!rgb) return 2;

  // Scale: use [bkg-2*sigma, bkg+6*sigma]
  double vmin = bkg - 2.0*sigma;
  double vmax = bkg + 6.0*sigma;
  double inv = (vmax > vmin) ? (1.0/(vmax - vmin)) : 1.0;

  for (int yy = 0; yy < h; yy++) {
    long y = ry1 + yy;
    for (int xx = 0; xx < w; xx++) {
      long x = rx1 + xx;
      double v = img[y*nx + x];
      double t = (v - vmin) * inv;
      if (t < 0) t = 0;
      if (t > 1) t = 1;
      uint8_t g = (uint8_t)lround(255.0 * t);
      size_t idx = (size_t)(yy*w + xx) * 3;
      rgb[idx+0] = g;
      rgb[idx+1] = g;
      rgb[idx+2] = g;
    }
  }

  // Color pixels above threshold
  double thr = bkg + snr_thresh * sigma;
  for (int yy = 0; yy < h; yy++) {
    long y = ry1 + yy;
    for (int xx = 0; xx < w; xx++) {
      long x = rx1 + xx;
      double v = img[y*nx + x];
      if (v > thr) {
        set_px(rgb, w, h, xx, yy, 255, 80, 80);
      }
    }
  }

  // Overlay markers if found
  if (det && det->found) {
    double gx0 = (p->pixel_origin == 0) ? p->goal_x : (p->goal_x - 1.0);
    double gy0 = (p->pixel_origin == 0) ? p->goal_y : (p->goal_y - 1.0);

    // convert user->0based for plotting
    double cx0 = (p->pixel_origin == 0) ? det->cx : (det->cx - 1.0);
    double cy0 = (p->pixel_origin == 0) ? det->cy : (det->cy - 1.0);
    double px0 = (p->pixel_origin == 0) ? det->peak_x : (det->peak_x - 1.0);
    double py0 = (p->pixel_origin == 0) ? det->peak_y : (det->peak_y - 1.0);

    int gx = (int)lround(gx0 - (double)rx1);
    int gy = (int)lround(gy0 - (double)ry1);
    int cx = (int)lround(cx0 - (double)rx1);
    int cy = (int)lround(cy0 - (double)ry1);
    int px = (int)lround(px0 - (double)rx1);
    int py = (int)lround(py0 - (double)ry1);

    draw_x(rgb, w, h, gx, gy, 9, 40, 200, 255);
    draw_circle(rgb, w, h, px, py, 12, 255, 220, 40);
    draw_plus(rgb, w, h, cx, cy, 9, 40, 255, 80);
    draw_arrow(rgb, w, h, cx, cy, gx, gy, 255, 255, 255);
  }

  FILE* fp = fopen(outpath, "wb");
  if (!fp) { free(rgb); return 3; }
  fprintf(fp, "P6\n%d %d\n255\n", w, h);
  fwrite(rgb, 1, (size_t)w * (size_t)h * 3, fp);
  fclose(fp);
  free(rgb);
  return 0;
}

// Move to IMAGE HDU by EXTNAME match. Returns 0 on success.
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

  return 4;
}

// Read 2D float image + header string from preferred EXTNAME (if set), else extnum.
static int read_fits_image_and_header(const char* path, const AcqParams* p,
                                      float** img_out, long* nx_out, long* ny_out,
                                      char** header_out, int* nkeys_out)
{
  fitsfile* fptr = NULL;
  int status = 0;

  if (fits_open_file(&fptr, path, READONLY, &status)) return status;

  // Choose HDU
  if (p->extname[0]) {
    int found_hdu = 0;
    int rc = move_to_image_hdu_by_extname(fptr, p->extname, &found_hdu, &status);
    if (rc == 0) {
      // already moved
    } else {
      if (p->verbose) fprintf(stderr, "WARNING: EXTNAME='%s' not found; falling back to extnum=%d\n", p->extname, p->extnum);
      status = 0;
      int hdutype = 0;
      if (fits_movabs_hdu(fptr, p->extnum + 1, &hdutype, &status)) {
        fits_close_file(fptr, &status);
        return status;
      }
    }
  } else {
    int hdutype = 0;
    if (fits_movabs_hdu(fptr, p->extnum + 1, &hdutype, &status)) {
      fits_close_file(fptr, &status);
      return status;
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

// Execute a TCS move (dra,ddec in arcsec).
static int tcs_move_arcsec(double dra_arcsec, double ddec_arcsec, const AcqParams* p)
{
  char cmd[512];
  snprintf(cmd, sizeof(cmd), "tcs native pt %.3f %.3f", dra_arcsec, ddec_arcsec);

  if (p->verbose || p->dry_run) fprintf(stderr, "TCS CMD: %s\n", cmd);
  if (p->dry_run) return 0;

  int rc = system(cmd);
  if (rc != 0) {
    fprintf(stderr, "WARNING: TCS command returned %d\n", rc);
    return 1;
  }
  return 0;
}

static int tcs_set_units_once(const AcqParams* p)
{
  if (!p->tcs_set_units) return 0;
  const char* cmd1 = "tcs native dra 'arcsec'";
  const char* cmd2 = "tcs native ddec 'arcsec'";
  if (p->verbose || p->dry_run) {
    fprintf(stderr, "TCS SET: %s\n", cmd1);
    fprintf(stderr, "TCS SET: %s\n", cmd2);
  }
  if (p->dry_run) return 0;
  (void)system(cmd1);
  (void)system(cmd2);
  return 0;
}

// Iterative Gaussian-windowed centroid around a starting point.
static void windowed_centroid(const float* img, long nx, long ny,
                              long sx1, long sx2, long sy1, long sy2,
                              double bkg, int hw, double sigma_w,
                              int maxiter, double eps,
                              double x_start, double y_start,
                              double* x_out, double* y_out,
                              double* flux_out, int* npixpos_out)
{
  double xc = x_start;
  double yc = y_start;
  if (sigma_w <= 0.2) sigma_w = 0.2;

  for (int it = 0; it < maxiter; it++) {
    long x0 = (long)floor(xc) - hw;
    long x1 = (long)floor(xc) + hw;
    long y0 = (long)floor(yc) - hw;
    long y1 = (long)floor(yc) + hw;

    if (x0 < sx1) x0 = sx1;
    if (y0 < sy1) y0 = sy1;
    if (x1 > sx2) x1 = sx2;
    if (y1 > sy2) y1 = sy2;

    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 > nx-1) x1 = nx-1;
    if (y1 > ny-1) y1 = ny-1;

    double sumW = 0.0, sumX = 0.0, sumY = 0.0;
    double flux = 0.0;
    int np = 0;

    for (long y = y0; y <= y1; y++) {
      for (long x = x0; x <= x1; x++) {
        double I = (double)img[y*nx + x] - bkg;
        if (I <= 0) continue;
        double dx = ((double)x - xc);
        double dy = ((double)y - yc);
        double w = exp(-0.5*(dx*dx + dy*dy)/(sigma_w*sigma_w));
        double ww = w * I;
        sumW += ww;
        sumX += ww * (double)x;
        sumY += ww * (double)y;
        flux += I;
        np++;
      }
    }

    if (sumW <= 0.0) break;

    double xn = sumX / sumW;
    double yn = sumY / sumW;
    double sh = hypot(xn - xc, yn - yc);
    xc = xn;
    yc = yn;

    if (it == maxiter-1 || sh < eps) {
      if (flux_out) *flux_out = flux;
      if (npixpos_out) *npixpos_out = np;
      break;
    }

    if (flux_out) *flux_out = flux;
    if (npixpos_out) *npixpos_out = np;
  }

  *x_out = xc;
  *y_out = yc;
}

// Robust star detection near goal.
static Detection detect_star(const float* img, long nx, long ny, const AcqParams* p)
{
  Detection d;
  memset(&d, 0, sizeof(d));

  // Background stats ROI
  long bgx1, bgx2, bgy1, bgy2;
  compute_roi_0based(nx, ny, p->pixel_origin,
                     p->bg_roi_mask, p->bg_x1, p->bg_x2, p->bg_y1, p->bg_y2,
                     &bgx1, &bgx2, &bgy1, &bgy2);

  double bkg = 0.0, sigma = 1.0;
  bg_sigma_sextractor_like(img, nx, ny, bgx1, bgx2, bgy1, bgy2, &bkg, &sigma);
  if (!isfinite(sigma) || sigma <= 0) sigma = 1.0;

  d.bkg = bkg;
  d.sigma = sigma;

  // Search ROI
  long sx1, sx2, sy1, sy2;
  if (p->search_roi_mask == 0) {
    sx1 = bgx1; sx2 = bgx2; sy1 = bgy1; sy2 = bgy2;
  } else {
    compute_roi_0based(nx, ny, p->pixel_origin,
                       p->search_roi_mask, p->search_x1, p->search_x2, p->search_y1, p->search_y2,
                       &sx1, &sx2, &sy1, &sy2);
  }

  // Goal in 0-based pixels
  const double goal_x0 = (p->pixel_origin == 0) ? p->goal_x : (p->goal_x - 1.0);
  const double goal_y0 = (p->pixel_origin == 0) ? p->goal_y : (p->goal_y - 1.0);

  // Candidate window (square) around goal
  long x0 = (long)floor(goal_x0 - p->max_dist_pix);
  long x1 = (long)ceil (goal_x0 + p->max_dist_pix);
  long y0 = (long)floor(goal_y0 - p->max_dist_pix);
  long y1 = (long)ceil (goal_y0 + p->max_dist_pix);

  if (x0 < sx1) x0 = sx1;
  if (x1 > sx2) x1 = sx2;
  if (y0 < sy1) y0 = sy1;
  if (y1 > sy2) y1 = sy2;

  // Keep margins for local-max neighborhood checks
  if (x0 < 1) x0 = 1;
  if (y0 < 1) y0 = 1;
  if (x1 > nx-2) x1 = nx-2;
  if (y1 > ny-2) y1 = ny-2;

  if (x1 <= x0 || y1 <= y0) {
    d.found = 0;
    return d;
  }

  d.cand_x0 = x0;
  d.cand_x1 = x1;
  d.cand_y0 = y0;
  d.cand_y1 = y1;

  // Extract background-subtracted patch for filtering
  int w = (int)(x1 - x0 + 1);
  int h = (int)(y1 - y0 + 1);
  float* patch = (float*)malloc((size_t)w * (size_t)h * sizeof(float));
  float* tmp   = (float*)malloc((size_t)w * (size_t)h * sizeof(float));
  float* filt  = (float*)malloc((size_t)w * (size_t)h * sizeof(float));
  if (!patch || !tmp || !filt) die("malloc patch/filter failed");

  for (int yy = 0; yy < h; yy++) {
    long y = y0 + yy;
    for (int xx = 0; xx < w; xx++) {
      long x = x0 + xx;
      patch[yy*w + xx] = (float)((double)img[y*nx + x] - bkg);
    }
  }

  int kr = 0;
  double* k = make_gaussian_kernel(p->filt_sigma_pix, &kr);
  double s2 = kernel_sum_sq(k, kr);
  double sigma_filt = sigma * s2; // because sqrt(sum(w^2)) = s2 for separable normalized kernel

  convolve_separable(patch, tmp, filt, w, h, k, kr);

  double thr_f = p->snr_thresh * sigma_filt;
  double thr_raw = bkg + p->snr_thresh * sigma;

  // Find best local maximum in filtered image
  float best_v = -1e30f;
  int best_x = -1, best_y = -1;
  for (int yy = 1; yy < h-1; yy++) {
    for (int xx = 1; xx < w-1; xx++) {
      float v = filt[yy*w + xx];
      if (v <= (float)thr_f) continue;

      // local max in 3x3 (filtered)
      int ismax = 1;
      for (int dy = -1; dy <= 1 && ismax; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
          if (dx == 0 && dy == 0) continue;
          if (filt[(yy+dy)*w + (xx+dx)] >= v) { ismax = 0; break; }
        }
      }
      if (!ismax) continue;

      // Must be within circular max_dist
      double gx = (double)(x0 + xx) - goal_x0;
      double gy = (double)(y0 + yy) - goal_y0;
      if (hypot(gx, gy) > p->max_dist_pix) continue;

      // Adjacent pixels above raw threshold (use raw image)
      int nadj = 0;
      long X = x0 + xx;
      long Y = y0 + yy;
      for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
          if (dx == 0 && dy == 0) continue;
          if ((double)img[(Y+dy)*nx + (X+dx)] > thr_raw) nadj++;
        }
      }
      if (nadj < p->min_adjacent) continue;

      if (v > best_v) {
        best_v = v;
        best_x = xx;
        best_y = yy;
      }
    }
  }

  if (best_x < 0) {
    d.found = 0;
    free(patch); free(tmp); free(filt); free(k);
    return d;
  }

  long peak_x0 = x0 + best_x;
  long peak_y0 = y0 + best_y;

  d.peak_val = (double)img[peak_y0*nx + peak_x0];
  d.peak_snr_raw = (d.peak_val - bkg) / sigma;

  // Windowed centroid around peak; clamp to search ROI
  double cx = (double)peak_x0;
  double cy = (double)peak_y0;
  double flux = 0.0;
  int npixpos = 0;

  windowed_centroid(img, nx, ny,
                    sx1, sx2, sy1, sy2,
                    bkg,
                    p->centroid_halfwin,
                    p->centroid_sigma_pix,
                    p->centroid_maxiter,
                    p->centroid_eps_pix,
                    cx, cy,
                    &cx, &cy,
                    &flux, &npixpos);

  d.snr_ap = (npixpos > 0) ? (flux / (sigma * sqrt((double)npixpos))) : 0.0;

  // Convert to user origin
  d.found = 1;
  d.peak_x = (p->pixel_origin == 0) ? (double)peak_x0 : (double)peak_x0 + 1.0;
  d.peak_y = (p->pixel_origin == 0) ? (double)peak_y0 : (double)peak_y0 + 1.0;
  d.cx     = (p->pixel_origin == 0) ? cx : cx + 1.0;
  d.cy     = (p->pixel_origin == 0) ? cy : cy + 1.0;

  // For reporting, use aperture-like SNR if it is valid, else peak
  d.snr_ap = isfinite(d.snr_ap) ? d.snr_ap : 0.0;

  free(patch); free(tmp); free(filt); free(k);
  return d;
}

static int stat_file_basic(const char* path, time_t* mtime_out, off_t* size_out)
{
  struct stat st;
  if (stat(path, &st) != 0) return 1;
  if (!S_ISREG(st.st_mode)) return 2;
  if (mtime_out) *mtime_out = st.st_mtime;
  if (size_out) *size_out = st.st_size;
  return 0;
}

// Returns:
//   0: success (new + stable)
//   1: not new
//   2+: error
static int read_if_new_and_stable(const char* path, time_t* last_mtime, off_t* last_size,
                                  const AcqParams* p,
                                  float** img_out, long* nx_out, long* ny_out,
                                  char** header_out, int* nkeys_out)
{
  time_t mt1=0, mt2=0;
  off_t  sz1=0, sz2=0;

  if (stat_file_basic(path, &mt1, &sz1) != 0) return 2;
  if (sz1 <= 0) return 2;

  if (*last_mtime != 0 && mt1 == *last_mtime && sz1 == *last_size) return 1;

  // stability check (avoid reading while being written)
  sleep_seconds(0.12);
  if (stat_file_basic(path, &mt2, &sz2) != 0) return 2;
  if (mt2 != mt1 || sz2 != sz1) return 2;

  int st = read_fits_image_and_header(path, p, img_out, nx_out, ny_out, header_out, nkeys_out);
  if (st) return 3;

  *last_mtime = mt1;
  *last_size  = sz1;
  return 0;
}

static FrameResult process_frame(const char* path, const AcqParams* p, int do_debug)
{
  FrameResult fr;
  memset(&fr, 0, sizeof(fr));

  float* img = NULL;
  long nx=0, ny=0;
  char* header = NULL;
  int nkeys = 0;

  int st = read_fits_image_and_header(path, p, &img, &nx, &ny, &header, &nkeys);
  if (st) {
    if (p->verbose) fprintf(stderr, "ERROR: CFITSIO read failed for %s (status=%d)\n", path, st);
    if (img) free(img);
    if (header) free(header);
    return fr;
  }

  Detection det = detect_star(img, nx, ny, p);
  fr.det = det;

  if (!det.found) {
    if (p->verbose) fprintf(stderr, "No suitable star detected near goal.\n");
    printf("NGPS_ACQ_RESULT found=0\n");
    free(img);
    if (header) free(header);
    return fr;
  }

  // pixel offsets (in user origin)
  fr.dx_pix = det.cx - p->goal_x;
  fr.dy_pix = det.cy - p->goal_y;

  // WCS
  struct wcsprm* wcs = NULL;
  int nwcs = 0;
  int wcs_stat = init_wcs_from_header(header, nkeys, &wcs, &nwcs);
  if (wcs_stat != 0) {
    if (p->verbose) fprintf(stderr, "ERROR: WCS parse failed (code=%d).\n", wcs_stat);
    printf("NGPS_ACQ_RESULT found=1 cx=%.6f cy=%.6f dx_pix=%.6f dy_pix=%.6f snr_ap=%.3f wcs_ok=0\n",
           det.cx, det.cy, fr.dx_pix, fr.dy_pix, det.snr_ap);

    // debug image still useful
    if (do_debug && p->debug_out[0]) {
      long bgx1,bgx2,bgy1,bgy2;
      compute_roi_0based(nx, ny, p->pixel_origin,
                         p->bg_roi_mask, p->bg_x1, p->bg_x2, p->bg_y1, p->bg_y2,
                         &bgx1, &bgx2, &bgy1, &bgy2);
      (void)write_debug_ppm(p->debug_out, img, nx, ny, bgx1, bgx2, bgy1, bgy2,
                            det.bkg, det.sigma, p->snr_thresh, &det, p);
    }

    wcsvfree(&nwcs, &wcs);
    free(img);
    if (header) free(header);
    return fr;
  }

  // Convert user->FITS 1-based for WCS
  double goal_x1 = (p->pixel_origin == 0) ? (p->goal_x + 1.0) : p->goal_x;
  double goal_y1 = (p->pixel_origin == 0) ? (p->goal_y + 1.0) : p->goal_y;
  double star_x1 = (p->pixel_origin == 0) ? (det.cx + 1.0)     : det.cx;
  double star_y1 = (p->pixel_origin == 0) ? (det.cy + 1.0)     : det.cy;

  double ra_goal=0, dec_goal=0, ra_star=0, dec_star=0;
  if (pix2world_wcs(&wcs[0], goal_x1, goal_y1, &ra_goal, &dec_goal) ||
      pix2world_wcs(&wcs[0], star_x1, star_y1, &ra_star, &dec_star)) {
    if (p->verbose) fprintf(stderr, "ERROR: WCS pix2world failed.\n");
    wcsvfree(&nwcs, &wcs);
    free(img);
    if (header) free(header);
    return fr;
  }

  fr.wcs_ok = 1;
  fr.ra_goal_deg = ra_goal; fr.dec_goal_deg = dec_goal;
  fr.ra_star_deg = ra_star; fr.dec_star_deg = dec_star;

  // Commanded offsets that move star onto goal: (star - goal)
  double dra_deg  = wrap_dra_deg(ra_star - ra_goal);
  double ddec_deg = (dec_star - dec_goal);

  double dra_arcsec = dra_deg * 3600.0;
  if (p->dra_use_cosdec) {
    double cosdec = cos(dec_goal * M_PI / 180.0);
    dra_arcsec *= cosdec;
  }
  double ddec_arcsec = ddec_deg * 3600.0;

  dra_arcsec *= (double)p->tcs_sign;
  ddec_arcsec *= (double)p->tcs_sign;

  fr.dra_cmd_arcsec = dra_arcsec;
  fr.ddec_cmd_arcsec = ddec_arcsec;
  fr.r_cmd_arcsec = hypot(dra_arcsec, ddec_arcsec);

  fr.ok = 1;

  // machine-readable output
  printf("NGPS_ACQ_RESULT found=1 cx=%.6f cy=%.6f dx_pix=%.6f dy_pix=%.6f dra_arcsec=%.6f ddec_arcsec=%.6f r_arcsec=%.6f snr_ap=%.3f peak_snr=%.3f bkg=%.3f sigma=%.3f wcs_ok=1\n",
         det.cx, det.cy, fr.dx_pix, fr.dy_pix,
         fr.dra_cmd_arcsec, fr.ddec_cmd_arcsec, fr.r_cmd_arcsec,
         det.snr_ap, det.peak_snr_raw, det.bkg, det.sigma);

  if (p->verbose) {
    fprintf(stderr, "Star: peak=(%.3f,%.3f) centroid=(%.3f,%.3f) dx=%.3f dy=%.3f pix | SNR_ap=%.2f peakSNR=%.2f\n",
            det.peak_x, det.peak_y, det.cx, det.cy, fr.dx_pix, fr.dy_pix, det.snr_ap, det.peak_snr_raw);
    fprintf(stderr, "TCS offsets: dra=%.3f\" ddec=%.3f\" (r=%.3f\")\n",
            fr.dra_cmd_arcsec, fr.ddec_cmd_arcsec, fr.r_cmd_arcsec);
  }

  if (do_debug && p->debug_out[0]) {
    long bgx1,bgx2,bgy1,bgy2;
    compute_roi_0based(nx, ny, p->pixel_origin,
                       p->bg_roi_mask, p->bg_x1, p->bg_x2, p->bg_y1, p->bg_y2,
                       &bgx1, &bgx2, &bgy1, &bgy2);
    int rc = write_debug_ppm(p->debug_out, img, nx, ny, bgx1, bgx2, bgy1, bgy2,
                             det.bkg, det.sigma, p->snr_thresh, &det, p);
    if (p->verbose && rc==0) fprintf(stderr, "Wrote debug overlay: %s\n", p->debug_out);
  }

  wcsvfree(&nwcs, &wcs);
  free(img);
  if (header) free(header);
  return fr;
}

static void usage(const char* argv0)
{
  fprintf(stderr,
    "Usage: %s --input FILE.fits --goal-x X --goal-y Y [options]\n"
    "\n"
    "Main mode: single FITS file (updated by camera). No directory mode.\n"
    "\n"
    "Core options:\n"
    "  --pixel-origin 0|1        Pixel origin for goal/ROI (default 0)\n"
    "  --max-dist PIX            Search radius around goal (default 200)\n"
    "  --snr S                   Detection threshold in sigma (default 8)\n"
    "  --min-adj N               Min adjacent raw pixels above threshold (default 4)\n"
    "\n"
    "Detection / centroiding:\n"
    "  --filt-sigma PIX          Gaussian smoothing sigma for detection (default 1.0)\n"
    "  --centroid-hw N           Centroid window half-width (default 6)\n"
    "  --centroid-sigma PIX      Gaussian window sigma (default 2.0)\n"
    "  --centroid-maxiter N      (default 10)\n"
    "\n"
    "FITS selection:\n"
    "  --extname NAME            Preferred image EXTNAME (default L). Use 'none' to disable\n"
    "  --extnum N                Fallback HDU (0=primary, 1=first ext...) (default 1)\n"
    "\n"
    "ROIs (inclusive bounds; same origin as goal):\n"
    "  Background stats ROI:\n"
    "    --bg-x1 N --bg-x2 N --bg-y1 N --bg-y2 N\n"
    "    (aliases: --roi-x1/--roi-x2/--roi-y1/--roi-y2)\n"
    "  Candidate search ROI (defaults to bg ROI):\n"
    "    --search-x1 N --search-x2 N --search-y1 N --search-y2 N\n"
    "\n"
    "Closed-loop acquisition wrapper:\n"
    "  --loop 0|1                Enable closed-loop mode (default 0)\n"
    "  --cadence-sec S           Seconds between accepted samples (default 4)\n"
    "  --prec-arcsec A           Required centroiding precision per axis (MAD->sigma) (default 0.1)\n"
    "  --goal-arcsec A           Stop when robust |offset| < A (default 0.1)\n"
    "  --max-samples N           Max samples gathered per move (default 10)\n"
    "  --min-samples N           Min samples before checking precision (default 3)\n"
    "  --max-cycles N            Max move cycles (default 50)\n"
    "  --gain G                  Multiply commanded move (default 1.0; try 0.8)\n"
    "\n"
    "TCS/WCS conventions:\n"
    "  --dra-cosdec 0|1          Use dra = dRA*cos(dec) (default 1)\n"
    "  --tcs-sign -1|1           Multiply commanded offsets by +/-1 (default 1)\n"
    "  --tcs-set-units 0|1       Run: tcs native dra 'arcsec' and ddec 'arcsec' once (default 1)\n"
    "\n"
    "Debug:\n"
    "  --debug 0|1               Write debug PPM overlay (default 0)\n"
    "  --debug-out FILE          Debug PPM path (default ngps_acq_debug.ppm)\n"
    "\n"
    "Other:\n"
    "  --dry-run 0|1             Do not call TCS (default 0)\n"
    "  --verbose 0|1             Verbose logging (default 1)\n",
    argv0);
}

static void set_defaults(AcqParams* p)
{
  memset(p, 0, sizeof(*p));
  snprintf(p->input, sizeof(p->input), "");

  p->goal_x = 0;
  p->goal_y = 0;
  p->pixel_origin = 0;

  p->max_dist_pix = 200.0;
  p->snr_thresh = 8.0;
  p->min_adjacent = 4;

  p->filt_sigma_pix = 1.0;

  p->centroid_halfwin = 6;
  p->centroid_sigma_pix = 2.0;
  p->centroid_maxiter = 10;
  p->centroid_eps_pix = 0.01;

  snprintf(p->extname, sizeof(p->extname), "L");
  p->extnum = 1;

  p->bg_roi_mask = 0;
  p->search_roi_mask = 0;

  p->loop = 0;
  p->cadence_sec = 4.0;
  p->max_samples = 10;
  p->min_samples = 3;
  p->prec_arcsec = 0.1;
  p->goal_arcsec = 0.1;
  p->max_cycles = 50;
  p->gain = 1.0;

  p->dra_use_cosdec = 1;
  p->tcs_sign = 1;
  p->tcs_set_units = 1;

  p->debug = 0;
  snprintf(p->debug_out, sizeof(p->debug_out), "ngps_acq_debug.ppm");

  p->dry_run = 0;
  p->verbose = 1;
}

static int parse_args(int argc, char** argv, AcqParams* p)
{
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--input") && i+1 < argc) {
      snprintf(p->input, sizeof(p->input), "%s", argv[++i]);
    } else if (!strcmp(argv[i], "--goal-x") && i+1 < argc) {
      p->goal_x = atof(argv[++i]);
    } else if (!strcmp(argv[i], "--goal-y") && i+1 < argc) {
      p->goal_y = atof(argv[++i]);
    } else if (!strcmp(argv[i], "--pixel-origin") && i+1 < argc) {
      p->pixel_origin = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "--max-dist") && i+1 < argc) {
      p->max_dist_pix = atof(argv[++i]);
    } else if (!strcmp(argv[i], "--snr") && i+1 < argc) {
      p->snr_thresh = atof(argv[++i]);
    } else if (!strcmp(argv[i], "--min-adj") && i+1 < argc) {
      p->min_adjacent = atoi(argv[++i]);

    } else if (!strcmp(argv[i], "--filt-sigma") && i+1 < argc) {
      p->filt_sigma_pix = atof(argv[++i]);
    } else if (!strcmp(argv[i], "--centroid-hw") && i+1 < argc) {
      p->centroid_halfwin = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "--centroid-sigma") && i+1 < argc) {
      p->centroid_sigma_pix = atof(argv[++i]);
    } else if (!strcmp(argv[i], "--centroid-maxiter") && i+1 < argc) {
      p->centroid_maxiter = atoi(argv[++i]);

    } else if (!strcmp(argv[i], "--extnum") && i+1 < argc) {
      p->extnum = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "--extname") && i+1 < argc) {
      snprintf(p->extname, sizeof(p->extname), "%s", argv[++i]);
      if (!strcasecmp(p->extname, "none")) p->extname[0] = '\0';

    } else if (!strcmp(argv[i], "--loop") && i+1 < argc) {
      p->loop = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "--cadence-sec") && i+1 < argc) {
      p->cadence_sec = atof(argv[++i]);
    } else if (!strcmp(argv[i], "--prec-arcsec") && i+1 < argc) {
      p->prec_arcsec = atof(argv[++i]);
    } else if (!strcmp(argv[i], "--goal-arcsec") && i+1 < argc) {
      p->goal_arcsec = atof(argv[++i]);
    } else if (!strcmp(argv[i], "--max-samples") && i+1 < argc) {
      p->max_samples = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "--min-samples") && i+1 < argc) {
      p->min_samples = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "--max-cycles") && i+1 < argc) {
      p->max_cycles = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "--gain") && i+1 < argc) {
      p->gain = atof(argv[++i]);

    } else if (!strcmp(argv[i], "--dra-cosdec") && i+1 < argc) {
      p->dra_use_cosdec = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "--tcs-sign") && i+1 < argc) {
      p->tcs_sign = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "--tcs-set-units") && i+1 < argc) {
      p->tcs_set_units = atoi(argv[++i]);

    } else if (!strcmp(argv[i], "--debug") && i+1 < argc) {
      p->debug = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "--debug-out") && i+1 < argc) {
      snprintf(p->debug_out, sizeof(p->debug_out), "%s", argv[++i]);

    } else if (!strcmp(argv[i], "--dry-run") && i+1 < argc) {
      p->dry_run = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "--verbose") && i+1 < argc) {
      p->verbose = atoi(argv[++i]);

    // Background ROI
    } else if (!strcmp(argv[i], "--bg-x1") && i+1 < argc) {
      p->bg_x1 = atol(argv[++i]); p->bg_roi_mask |= ROI_X1_SET;
    } else if (!strcmp(argv[i], "--bg-x2") && i+1 < argc) {
      p->bg_x2 = atol(argv[++i]); p->bg_roi_mask |= ROI_X2_SET;
    } else if (!strcmp(argv[i], "--bg-y1") && i+1 < argc) {
      p->bg_y1 = atol(argv[++i]); p->bg_roi_mask |= ROI_Y1_SET;
    } else if (!strcmp(argv[i], "--bg-y2") && i+1 < argc) {
      p->bg_y2 = atol(argv[++i]); p->bg_roi_mask |= ROI_Y2_SET;

    // Alias roi-* for bg ROI
    } else if (!strcmp(argv[i], "--roi-x1") && i+1 < argc) {
      p->bg_x1 = atol(argv[++i]); p->bg_roi_mask |= ROI_X1_SET;
    } else if (!strcmp(argv[i], "--roi-x2") && i+1 < argc) {
      p->bg_x2 = atol(argv[++i]); p->bg_roi_mask |= ROI_X2_SET;
    } else if (!strcmp(argv[i], "--roi-y1") && i+1 < argc) {
      p->bg_y1 = atol(argv[++i]); p->bg_roi_mask |= ROI_Y1_SET;
    } else if (!strcmp(argv[i], "--roi-y2") && i+1 < argc) {
      p->bg_y2 = atol(argv[++i]); p->bg_roi_mask |= ROI_Y2_SET;

    // Search ROI
    } else if (!strcmp(argv[i], "--search-x1") && i+1 < argc) {
      p->search_x1 = atol(argv[++i]); p->search_roi_mask |= ROI_X1_SET;
    } else if (!strcmp(argv[i], "--search-x2") && i+1 < argc) {
      p->search_x2 = atol(argv[++i]); p->search_roi_mask |= ROI_X2_SET;
    } else if (!strcmp(argv[i], "--search-y1") && i+1 < argc) {
      p->search_y1 = atol(argv[++i]); p->search_roi_mask |= ROI_Y1_SET;
    } else if (!strcmp(argv[i], "--search-y2") && i+1 < argc) {
      p->search_y2 = atol(argv[++i]); p->search_roi_mask |= ROI_Y2_SET;

    } else if (!strcmp(argv[i], "--help")) {
      return 0;
    } else {
      fprintf(stderr, "Unknown/invalid arg: %s\n", argv[i]);
      return -1;
    }
  }

  if (p->input[0] == '\0') {
    fprintf(stderr, "You must provide --input FILE.fits\n");
    return -1;
  }
  if (p->goal_x == 0 && p->goal_y == 0) {
    fprintf(stderr, "You must provide --goal-x and --goal-y\n");
    return -1;
  }
  if (!(p->pixel_origin == 0 || p->pixel_origin == 1)) {
    fprintf(stderr, "--pixel-origin must be 0 or 1\n");
    return -1;
  }
  if (p->max_dist_pix < 3) p->max_dist_pix = 3;
  if (p->snr_thresh < 1) p->snr_thresh = 1;
  if (p->min_adjacent < 0) p->min_adjacent = 0;
  if (p->centroid_halfwin < 2) p->centroid_halfwin = 2;
  if (p->centroid_maxiter < 1) p->centroid_maxiter = 1;
  if (p->cadence_sec < 0.1) p->cadence_sec = 0.1;
  if (p->max_samples < 1) p->max_samples = 1;
  if (p->min_samples < 1) p->min_samples = 1;
  if (p->min_samples > p->max_samples) p->min_samples = p->max_samples;
  if (p->max_cycles < 1) p->max_cycles = 1;
  if (p->gain <= 0) p->gain = 1.0;
  if (p->gain > 1.5) p->gain = 1.5;
  if (!(p->tcs_sign == 1 || p->tcs_sign == -1)) p->tcs_sign = 1;

  return 1;
}

static int run_one_shot(const AcqParams* p)
{
  (void)tcs_set_units_once(p);
  FrameResult fr = process_frame(p->input, p, p->debug);
  if (!fr.ok) return 2;

  if (fr.r_cmd_arcsec <= p->goal_arcsec) {
    if (p->verbose) fprintf(stderr, "Within goal threshold (%.3f\") - no move.\n", p->goal_arcsec);
    return 0;
  }

  // One-shot move
  (void)tcs_move_arcsec(p->gain * fr.dra_cmd_arcsec, p->gain * fr.ddec_cmd_arcsec, p);
  return 0;
}

static int run_closed_loop(const AcqParams* p)
{
  (void)tcs_set_units_once(p);

  time_t last_mtime = 0;
  off_t  last_size  = 0;

  if (p->verbose) {
    fprintf(stderr,
            "Closed-loop acquisition:\n"
            "  cadence=%.2fs max_samples=%d min_samples=%d prec=%.3f\" goal=%.3f\" gain=%.2f max_cycles=%d\n",
            p->cadence_sec, p->max_samples, p->min_samples, p->prec_arcsec, p->goal_arcsec, p->gain, p->max_cycles);
  }

  for (int cycle = 1; cycle <= p->max_cycles && !g_stop; cycle++) {
    if (p->verbose) fprintf(stderr, "\n=== Cycle %d/%d: gather offsets (no move yet) ===\n", cycle, p->max_cycles);

    double* dra = (double*)calloc((size_t)p->max_samples, sizeof(double));
    double* ddec = (double*)calloc((size_t)p->max_samples, sizeof(double));
    double* r = (double*)calloc((size_t)p->max_samples, sizeof(double));
    if (!dra || !ddec || !r) die("calloc samples failed");

    int ns = 0;
    while (ns < p->max_samples && !g_stop) {
      // Wait for a new stable frame
      float* img = NULL; long nx=0, ny=0; char* header=NULL; int nkeys=0;
      int rr = read_if_new_and_stable(p->input, &last_mtime, &last_size, p, &img, &nx, &ny, &header, &nkeys);
      if (rr == 1) {
        sleep_seconds(0.10);
        continue;
      } else if (rr != 0) {
        if (p->verbose) fprintf(stderr, "WARNING: could not read new stable frame (rr=%d).\n", rr);
        sleep_seconds(0.20);
        continue;
      }

      // We already read the file; reuse the data path by temporarily writing to a temp file is silly.
      // Instead, process on the already-read arrays would require refactoring.
      // So: free and re-call process_frame() (CFITSIO read). The stability gate ensures consistency.
      free(img);
      if (header) free(header);

      FrameResult fr = process_frame(p->input, p, p->debug);
      if (!fr.ok) {
        if (p->verbose) fprintf(stderr, "Sample rejected (no star or WCS).\n");
        sleep_seconds(p->cadence_sec);
        continue;
      }

      dra[ns] = fr.dra_cmd_arcsec;
      ddec[ns] = fr.ddec_cmd_arcsec;
      r[ns] = fr.r_cmd_arcsec;
      ns++;

      // Print sample summary
      if (p->verbose) {
        fprintf(stderr, "Sample %d/%d: dra=%.3f\" ddec=%.3f\" r=%.3f\"\n",
                ns, p->max_samples, fr.dra_cmd_arcsec, fr.ddec_cmd_arcsec, fr.r_cmd_arcsec);
      }

      // Evaluate precision after min_samples
      if (ns >= p->min_samples) {
        double* tmpa = (double*)malloc((size_t)ns*sizeof(double));
        double* tmpd = (double*)malloc((size_t)ns*sizeof(double));
        if (!tmpa || !tmpd) die("malloc tmp med failed");
        memcpy(tmpa, dra, (size_t)ns*sizeof(double));
        memcpy(tmpd, ddec, (size_t)ns*sizeof(double));
        double med_a = median_of_doubles(tmpa, ns);
        double med_d = median_of_doubles(tmpd, ns);
        free(tmpa); free(tmpd);

        double sig_a = mad_sigma_of_doubles(dra, ns, med_a);
        double sig_d = mad_sigma_of_doubles(ddec, ns, med_d);

        if (p->verbose) {
          fprintf(stderr, "  Robust scatter: sig_dra=%.3f\" sig_ddec=%.3f\" (target < %.3f\")\n",
                  sig_a, sig_d, p->prec_arcsec);
        }

        if (sig_a < p->prec_arcsec && sig_d < p->prec_arcsec) {
          if (p->verbose) fprintf(stderr, "  Precision requirement met; stop gathering.\n");
          break;
        }
      }

      sleep_seconds(p->cadence_sec);
    }

    if (ns <= 0) {
      fprintf(stderr, "No valid samples collected in cycle %d; continuing.\n", cycle);
      free(dra); free(ddec); free(r);
      continue;
    }

    // Robust central value (median)
    double* tmpa = (double*)malloc((size_t)ns*sizeof(double));
    double* tmpd = (double*)malloc((size_t)ns*sizeof(double));
    if (!tmpa || !tmpd) die("malloc tmp med2 failed");
    memcpy(tmpa, dra, (size_t)ns*sizeof(double));
    memcpy(tmpd, ddec, (size_t)ns*sizeof(double));
    double med_a = median_of_doubles(tmpa, ns);
    double med_d = median_of_doubles(tmpd, ns);
    free(tmpa); free(tmpd);

    double rmed = hypot(med_a, med_d);

    if (p->verbose) {
      fprintf(stderr, "Robust offsets (median of %d): dra=%.3f\" ddec=%.3f\" r=%.3f\"\n", ns, med_a, med_d, rmed);
    }

    // Stop if converged
    if (rmed <= p->goal_arcsec) {
      if (p->verbose) fprintf(stderr, "Converged: r <= %.3f\".\n", p->goal_arcsec);
      free(dra); free(ddec); free(r);
      return 0;
    }

    // Command a move
    double dra_move = p->gain * med_a;
    double ddec_move = p->gain * med_d;

    if (p->verbose) {
      fprintf(stderr, "Issuing move (gain=%.2f): dra=%.3f\" ddec=%.3f\"\n", p->gain, dra_move, ddec_move);
    }

    (void)tcs_move_arcsec(dra_move, ddec_move, p);

    free(dra); free(ddec); free(r);

    // Let system settle a bit; also allow new frames to arrive
    sleep_seconds(p->cadence_sec);
  }

  if (g_stop) {
    fprintf(stderr, "Stopped by user (Ctrl+C).\n");
    return 0;
  }

  fprintf(stderr, "Reached max cycles (%d) without converging.\n", p->max_cycles);
  return 1;
}

int main(int argc, char** argv)
{
  signal(SIGINT, on_sigint);

  AcqParams p;
  set_defaults(&p);

  int pr = parse_args(argc, argv, &p);
  if (pr <= 0) { usage(argv[0]); return (pr == 0) ? 0 : 4; }

  // Basic sanity that file exists
  {
    time_t mt=0; off_t sz=0;
    if (stat_file_basic(p.input, &mt, &sz) != 0) {
      fprintf(stderr, "ERROR: --input is not a readable regular file: %s\n", p.input);
      return 4;
    }
  }

  if (p.verbose) {
    fprintf(stderr,
            "NGPS acquisition starting:\n"
            "  input=%s goal=(%.3f,%.3f) origin=%d max_dist=%.1f snr=%.1f min_adj=%d\n"
            "  filt_sigma=%.2f centroid_hw=%d centroid_sigma=%.2f extname=%s extnum=%d\n"
            "  dra_cosdec=%d tcs_sign=%d tcs_set_units=%d\n"
            "  loop=%d cadence=%.2f goal_arcsec=%.3f prec_arcsec=%.3f gain=%.2f\n"
            "  debug=%d debug_out=%s dry_run=%d\n",
            p.input, p.goal_x, p.goal_y, p.pixel_origin, p.max_dist_pix, p.snr_thresh, p.min_adjacent,
            p.filt_sigma_pix, p.centroid_halfwin, p.centroid_sigma_pix,
            (p.extname[0] ? p.extname : "(none)"), p.extnum,
            p.dra_use_cosdec, p.tcs_sign, p.tcs_set_units,
            p.loop, p.cadence_sec, p.goal_arcsec, p.prec_arcsec, p.gain,
            p.debug, p.debug_out, p.dry_run);
  }

  if (!p.loop) return run_one_shot(&p);
  return run_closed_loop(&p);
}
