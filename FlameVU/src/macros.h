#ifndef _MACROS_H_
#define _MACROS_H_

#define PI 3.14159265358


#define SAFE_DELETE(p)           do { delete (p); (p) = nullptr; } while(0)
#define SAFE_DELETE_ARRAY(p)     do { if(p) { delete[] (p); (p) = nullptr; } } while(0)
#define SAFE_FREE(p)             do { if(p) { free(p); (p) = nullptr; } } while(0)
#define SAFE_RELEASE(p)          do { if(p) { (p)->release(); } } while(0)
#define SAFE_RELEASE_NULL(p)     do { if(p) { (p)->release(); (p) = nullptr; } } while(0)
#define SAFE_RETAIN(p)           do { if(p) { (p)->retain(); } } while(0)
#define BREAK_IF(cond)           if(cond) break
#define SAFE_DELETE_DEVICE_ARRAY(p)	do { if(p) { checkCudaErrors(cudaFree(p)); } } while(0)

#endif // _MACROS_H_