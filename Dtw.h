#include <vector>
#include <algorithm>
#include <limits>
#include <tuple>
using std::vector;
using std::tuple;


tuple<int, int, double> findmin(vector< vector<double> >&D, int i, int j, int m, int n){
    if (D[i-1][j] <= D[i-1][j-1] && D[i-1][j]<= D[i][j-1]){
      return std::make_tuple(i-1, j, D[i-1][j]);
    }
    else if (D[i-1][j-1] <= D[i-1][j] && D[i-1][j-1] <= D[i][j-1]){
      return std::make_tuple(i-1, j-1, D[i-1][j-1]);
    }
    else
    {
      return std::make_tuple(i, j-1, D[i][j-1]);
    }
    
}
void forward(tensorflow::OpKernelContext* context, tensorflow::Tensor *p, 
    double *l, const double *d, int m, int n){
  vector< vector<double> > D(m+1);
  vector< int > px, py;
  
  vector< vector<int> > Px(m+1);
  for(int i=0;i<m+1;i++) Px[i].resize(n+1);
  for(int i=0;i<m+1;i++){
      for(int j=0;j<n+1;j++){
        Px[i][j] = -1;
      }
  }

  vector< vector<int> > Py(m+1);
  for(int i=0;i<m+1;i++) Py[i].resize(n+1);
  for(int i=0;i<m+1;i++){
      for(int j=0;j<n+1;j++){
        Py[i][j] = -1;
      }
  }

  for(int i=0;i<m+1;i++) D[i].resize(n+1);
  for(int i=0;i<m+1;i++){
      D[i][0] = std::numeric_limits<double>::infinity();
  }
  for(int i=0;i<n+1;i++){
      D[0][i] = std::numeric_limits<double>::infinity();
  }
  D[0][0] = 0.0;
  for(int i=1;i<m+1;i++){
    for(int j=1;j<n+1;j++){
      auto t = findmin(D, i, j, m, n);
      D[i][j] = d[(i-1)*n+j-1] + std::get<2>(t);
      Px[i][j] = std::get<0>(t);Py[i][j] = std::get<1>(t);
    }
  }
  *l = D[m][n];

  int k = 0;
  int i = m, j = n, ii, jj;
  while (true){
    if (Px[i][j]<=0) break;
    px.push_back(Px[i][j]); py.push_back(Py[i][j]);
    ii = Px[i][j];
    jj = Py[i][j]; 
    i = ii; j = jj;
  }

  tensorflow::TensorShape p_shape({static_cast<long long>(px.size()),2});
  OP_REQUIRES_OK(context, context->allocate_output(1, p_shape, &p));
  auto p_tensor = p->flat<int>().data();
  for(int i=0;i<px.size();i++){
    p_tensor[2*i] = px[i]; p_tensor[2*i+1] = py[i];
  }

  // for(int i=0;i<px.size();i++){
  //   printf("(%d, %d) --> ", px[i], py[i]);
  // }
  // printf("\n");
  // printf("*************\n");

  // for(int i=0;i<m+1;i++){
  //   for(int j=0;j<n+1;j++){
  //     printf("%f ", D[i][j]);
  //   }
  //   printf("\n");
  // }
}

void backward(
  double *grad_d, const double *grad_l,
  const int *p, int np, const double *l, const double *d, int m, int n){
    for(int i=0;i<m*n;i++){grad_d[i]=0.0;}
    for(int k=0;k<np;k++){
      int i = p[2*k]; int j = p[2*k+1];
      grad_d[(i-1)*n+j-1] = grad_l[0]; 
    }
    grad_d[m*n-1] = grad_l[0];
}