#include <cublasLt.h>
#include <cuda_runtime_api.h>
#include <vector>
static cublasLtHandle_t h;
static cublasLtMatmulDesc_t op;
static cublasLtMatrixLayout_t a,b,c;
static std::vector<cublasLtMatmulHeuristicResult_t> choices;
extern "C" int configure(int m,int n,int k,size_t workspace){
 if(h){cublasLtMatrixLayoutDestroy(a);cublasLtMatrixLayoutDestroy(b);cublasLtMatrixLayoutDestroy(c);cublasLtMatmulDescDestroy(op);cublasLtDestroy(h);}
 if(cublasLtCreate(&h))return -1;
 cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F);
 cublasOperation_t t=CUBLAS_OP_T;cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&t,sizeof(t));
 cublasLtMatrixLayoutCreate(&a,CUDA_R_16BF,k,n,k);
 cublasLtMatrixLayoutCreate(&b,CUDA_R_16BF,k,m,k);
 cublasLtMatrixLayoutCreate(&c,CUDA_R_16BF,n,m,n);
 cublasLtMatmulPreference_t p;cublasLtMatmulPreferenceCreate(&p);
 cublasLtMatmulPreferenceSetAttribute(p,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace,sizeof(workspace));
 choices.resize(32);int count=0;int status=cublasLtMatmulAlgoGetHeuristic(h,op,a,b,c,c,p,32,choices.data(),&count);
 cublasLtMatmulPreferenceDestroy(p);if(status)return -100-status;choices.resize(count);return count;
}
extern "C" int run(int i,void* x,void* w,void* y,void* scratch,size_t size,void* stream){
 if(i<0||i>=(int)choices.size())return -1;float alpha=1,beta=0;
 return cublasLtMatmul(h,op,&alpha,w,a,x,b,&beta,y,c,y,c,&choices[i].algo,scratch,size,(cudaStream_t)stream);
}
extern "C" int algorithm_id(int i){int id=-1;size_t n=0;cublasLtMatmulAlgoConfigGetAttribute(&choices[i].algo,CUBLASLT_ALGO_CONFIG_ID,&id,sizeof(id),&n);return id;}
