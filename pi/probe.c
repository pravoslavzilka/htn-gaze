#include <camera/camera_api.h>
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>

static int g_got;

static void dump(const char *label, const void *p, size_t n)
{
    const unsigned char *b = p;
    printf("%s (%zu bytes):\n", label, n);
    for (size_t i = 0; i < n; i++) {
        printf("%02x%s", b[i], ((i + 1) % 16 == 0 || i + 1 == n) ? "\n" : " ");
    }
}

static void cb(camera_handle_t handle, camera_buffer_t *buffer, void *arg)
{
    (void)handle;
    (void)arg;
    if (g_got || buffer == NULL) {
        return;
    }
    g_got = 1;
    printf("callback sizeof(camera_buffer_t)=%zu frametype_field=%d framebuf=%p\n",
           sizeof(camera_buffer_t), (int)buffer->frametype, (void *)buffer->framebuf);
    dump("camera_buffer_t", buffer, sizeof(*buffer) < 160 ? sizeof(*buffer) : 160);
}

int main(void)
{
    camera_handle_t handle = CAMERA_HANDLE_INVALID;
    int err;

    err = camera_open(CAMERA_UNIT_1, CAMERA_MODE_RO | CAMERA_MODE_PWRITE, &handle);
    printf("camera_open err=%d handle=%p\n", err, (void *)handle);
    if (err != CAMERA_EOK) {
        return 1;
    }

    err = camera_start_viewfinder(handle, cb, NULL, NULL);
    printf("start_viewfinder err=%d\n", err);
    for (int i = 0; i < 50 && !g_got; i++) {
        usleep(100000);
    }
    if (!g_got) {
        printf("no callback received\n");
    }
    camera_stop_viewfinder(handle);
    camera_close(handle);
    return 0;
}
