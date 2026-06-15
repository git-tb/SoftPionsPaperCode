#include <functional>

#include "cuba.h"

/**
 * @file integrate_nested.h
 * @author Tobias Bruschke
 * @brief Integrate a scalar function over a non-rectangular 2D domain using CUBA
 * @version 0.1
 * @date 2026-06-15
 * 
 * @copyright Copyright (c) 2026
 * 
 */

struct INTEGRATE_NESTED_ARGS   {
   const double& a1, b1;
   const std::function<double(double)>& a2, b2;
   const std::function<double(double, double)>& f;
};

/**
 * @brief int_a1^b1 dx1 int_a2(x1)^b2(x1) dx2 f(x1,x2)
 * 
 */
void integrate_nested(
   const double& a1, const double& b1,
   const std::function<double(double)>& a2, const std::function<double(double)>& b2,
   const std::function<double(double, double)>& f,
   double& result, double& error, double& prob,
   double epsrel, double epsabs, int iterations
)  {
   
   auto integrand_nested = [](const int* ndim, const double* x, const int* ncomp, double* fval, void* userdata) -> int
   {
      const INTEGRATE_NESTED_ARGS* args = static_cast<const INTEGRATE_NESTED_ARGS*>(userdata);

      const double x1 = args->a1 + (args->b1 - args->a1) * x[0]; // Map [0,1] to [a1,b1]
      const double a2_x1 = args->a2(x1);
      const double b2_x1 = args->b2(x1);
      const double x2 = a2_x1 + (b2_x1 - a2_x1) * x[1]; // Map [0,1] to [a2(x1), b2(x1)]
       
      const double jac = (args->b1 - args->a1) * (b2_x1 - a2_x1);

      fval[0] = args->f(x1, x2) * jac;
      return 0;
   };

   INTEGRATE_NESTED_ARGS args{a1, b1, a2, b2, f};

   const int   NDIM(2),
               NCOMP(1),
               NVEC(1),
               FLAGS(0),
               MINEVAL(0),
               MAXEVAL(iterations),
               KEY(0);
   void* USERDATA(&args);
   void* SPIN(NULL);
   const double EPSREL(epsrel), EPSABS(epsabs);
   const char* STATEFILE(NULL);

   int nregions, neval, fail;
   double integral[NCOMP], error_[NCOMP], prob_[NCOMP];
   Cuhre(NDIM, NCOMP, integrand_nested, USERDATA, NVEC,
         EPSREL, EPSABS, FLAGS,
         MINEVAL, MAXEVAL, KEY,
         STATEFILE, SPIN,
         &nregions, &neval, &fail, integral, error_, prob_);

   result = integral[0];
   error = error_[0];
   prob = prob_[0];

   return;
}