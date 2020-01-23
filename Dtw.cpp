#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<cmath>
#include "Dtw.h"

#ifdef USE_GPU
#include "tensorflow/core/util/gpu_kernel_helper.h"
namespace tensorflow{
  typedef Eigen::GpuDevice GPUDevice;
  void forwardGPU(const GPUDevice &d);
  void backwardGPU(const GPUDevice &d);
}
#endif

using namespace tensorflow;

REGISTER_OP("Dtw")

.Input("d : double")
.Output("l : double")
.Output("p : int32")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    
        shape_inference::ShapeHandle d_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &d_shape));

        c->set_output(0, c->Scalar());
        c->set_output(1, c->Matrix(-1,2));
    return Status::OK();
  });

REGISTER_OP("DtwGrad")

.Input("grad_l : double")
.Input("l : double")
.Input("p : int32")
.Input("d : double")
.Output("grad_d : double");


class DtwOp : public OpKernel {
private:
  
public:
  explicit DtwOp(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(1, context->num_inputs());
    
    
    const Tensor& d = context->input(0);
    
    
    const TensorShape& d_shape = d.shape();
    
    
    DCHECK_EQ(d_shape.dims(), 2);

    // extra check
        
    // create output shape
    int m = d_shape.dim_size(0), n = d_shape.dim_size(1);
    TensorShape l_shape({});
            
    // create output tensor
    
    Tensor* l = NULL;
    Tensor* p;
    OP_REQUIRES_OK(context, context->allocate_output(0, l_shape, &l));
    
    // get the corresponding Eigen tensors for data access
    
    auto d_tensor = d.flat<double>().data();
    auto l_tensor = l->flat<double>().data();

    // implement your forward function here 

    // TODO:
    forward(context, p, l_tensor, d_tensor, m, n);

  }
};
REGISTER_KERNEL_BUILDER(Name("Dtw").Device(DEVICE_CPU), DtwOp);



class DtwGradOp : public OpKernel {
private:
  
public:
  explicit DtwGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    const Tensor& grad_l = context->input(0);
    const Tensor& l = context->input(1);
    const Tensor& p = context->input(2);
    const Tensor& d = context->input(3);
    
    
    const TensorShape& grad_l_shape = grad_l.shape();
    const TensorShape& l_shape = l.shape();
    const TensorShape& p_shape = p.shape();
    const TensorShape& d_shape = d.shape();
    
    
    DCHECK_EQ(grad_l_shape.dims(), 0);
    DCHECK_EQ(l_shape.dims(), 0);
    DCHECK_EQ(p_shape.dims(), 2);
    DCHECK_EQ(d_shape.dims(), 2);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    int m = d_shape.dim_size(0), n = d_shape.dim_size(1);
    int np = p_shape.dim_size(0);

    TensorShape grad_d_shape(d_shape);
            
    // create output tensor
    
    Tensor* grad_d = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_d_shape, &grad_d));
    
    // get the corresponding Eigen tensors for data access
    
    auto d_tensor = d.flat<double>().data();
    auto grad_l_tensor = grad_l.flat<double>().data();
    auto l_tensor = l.flat<double>().data();
    auto p_tensor = p.flat<int32>().data();
    auto grad_d_tensor = grad_d->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    backward(grad_d_tensor, grad_l_tensor, p_tensor, np, l_tensor, d_tensor, m, n);
    
  }
};
REGISTER_KERNEL_BUILDER(Name("DtwGrad").Device(DEVICE_CPU), DtwGradOp);


/***************************************************************************************
**********************            GPU Operator            ******************************
***************************************************************************************/


#ifdef USE_GPU
class DtwOpGPU : public OpKernel {
private:
  
public:
  explicit DtwOpGPU(OpKernelConstruction* context) : OpKernel(context) {

  }

  void Compute(OpKernelContext* context) override {    
    DCHECK_EQ(1, context->num_inputs());
    
    
    const Tensor& d = context->input(0);
    
    
    const TensorShape& d_shape = d.shape();
    
    
    DCHECK_EQ(d_shape.dims(), 2);

    // extra check
        
    // create output shape
    
    TensorShape l_shape({});
    TensorShape p_shape({-1,2});
            
    // create output tensor
    
    Tensor* l = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, l_shape, &l));
    Tensor* p = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, p_shape, &p));
    
    // get the corresponding Eigen tensors for data access
    
    auto d_tensor = d.flat<double>().data();
    auto l_tensor = l->flat<double>().data();
    auto p_tensor = p->flat<int32>().data();   

    // implement your forward function here 

    // TODO:

  }
};
REGISTER_KERNEL_BUILDER(Name("Dtw").Device(DEVICE_GPU), DtwOpGPU);

class DtwGradOpGPU : public OpKernel {
private:
  
public:
  explicit DtwGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    
    
    const Tensor& grad_l = context->input(0);
    const Tensor& grad_p = context->input(1);
    const Tensor& l = context->input(2);
    const Tensor& p = context->input(3);
    const Tensor& d = context->input(4);
    
    
    const TensorShape& grad_l_shape = grad_l.shape();
    const TensorShape& grad_p_shape = grad_p.shape();
    const TensorShape& l_shape = l.shape();
    const TensorShape& p_shape = p.shape();
    const TensorShape& d_shape = d.shape();
    
    
    DCHECK_EQ(grad_l_shape.dims(), 0);
    DCHECK_EQ(grad_p_shape.dims(), 2);
    DCHECK_EQ(l_shape.dims(), 0);
    DCHECK_EQ(p_shape.dims(), 2);
    DCHECK_EQ(d_shape.dims(), 2);

    // extra check
    // int m = Example.dim_size(0);
        
    // create output shape
    
    TensorShape grad_d_shape(d_shape);
            
    // create output tensor
    
    Tensor* grad_d = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_d_shape, &grad_d));
    
    // get the corresponding Eigen tensors for data access
    
    auto d_tensor = d.flat<double>().data();
    auto grad_l_tensor = grad_l.flat<double>().data();
    auto grad_p_tensor = grad_p.flat<int32>().data();
    auto l_tensor = l.flat<double>().data();
    auto p_tensor = p.flat<int32>().data();
    auto grad_d_tensor = grad_d->flat<double>().data();   

    // implement your backward function here 

    // TODO:
    
  }
};
REGISTER_KERNEL_BUILDER(Name("DtwGrad").Device(DEVICE_GPU), DtwGradOpGPU);

#endif