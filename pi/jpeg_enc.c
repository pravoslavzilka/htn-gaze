#include "jpeg_enc.h"

#include <stdlib.h>
#include <string.h>

/* libturbojpeg is on the QNX image; headers are not. Call the C ABI directly. */
typedef void *tjhandle;
enum { TJPF_RGB = 0 };
enum { TJFLAG_FASTDCT = 2048 };

#ifdef __cplusplus
extern "C" {
#endif
extern tjhandle tjInitCompress(void);
extern int tjCompress2(tjhandle handle, const unsigned char *srcBuf, int width, int pitch,
                       int height, int pixelFormat, unsigned char **jpegBuf,
                       unsigned long *jpegSize, int jpegSubsamp, int jpegQual, int flags);
extern int tjDestroy(tjhandle handle);
extern void tjFree(unsigned char *buffer);
#ifdef __cplusplus
}
#endif

size_t jpeg_encode_rgb(const uint8_t *rgb, int width, int height, int quality,
                       uint8_t *out, size_t out_cap)
{
    tjhandle h;
    unsigned char *jbuf = NULL;
    unsigned long jsize = 0;
    int q;
    size_t n;

    if (rgb == NULL || out == NULL || width < 8 || height < 8 || out_cap < 128) {
        return 0;
    }
    q = quality;
    if (q < 20) {
        q = 20;
    }
    if (q > 95) {
        q = 95;
    }

    h = tjInitCompress();
    if (h == NULL) {
        return 0;
    }
    if (tjCompress2(h, rgb, width, 0, height, TJPF_RGB, &jbuf, &jsize, 2 /* TJSAMP_420 */, q,
                    TJFLAG_FASTDCT) != 0 ||
        jbuf == NULL || jsize == 0 || jsize > out_cap) {
        if (jbuf) {
            tjFree(jbuf);
        }
        tjDestroy(h);
        return 0;
    }
    memcpy(out, jbuf, (size_t)jsize);
    n = (size_t)jsize;
    tjFree(jbuf);
    tjDestroy(h);
    return n;
}
