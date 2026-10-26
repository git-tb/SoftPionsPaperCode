#include <iostream>
#include <cmath>
#include <complex>
#include <functional>
#include <boost/math/interpolators/pchip.hpp>
#include <boost/program_options.hpp>
#include <filesystem>
#include <ctime>   /// std::time, std::local_time
#include <iomanip> /// std::put_time

#include <algorithm>

#include "savedata.h"       /// write to file
#include "filereader.h"     /// readcsv
#include "constants.h"      /// GeV to inverse fm
#include "cuba.h"

#include "debugmsg.h"

/*
    decayV4 - same decay map as decayV2/decayV3, with the inner (v) integration
    variable substituted so that BOTH endpoint singularities are removed
    analytically.

    WHY. The integrand of decayformula.tex,

        t/(w_Tq w_Tp) * primespec(tQ) * heavyside(u-1)/sqrt(u^2-1) * 1/sqrt(1-v^2),

    is inverse-square-root singular at BOTH ends of the v-integration:
    1/sqrt(u^2-1) ~ (v-v1)^(-1/2) at the lower limit v1, and 1/sqrt(1-v^2) ~
    (1-v)^(-1/2) at the upper limit v=1. Cuhre samples the face v == 1 exactly,
    where the second factor is +inf; Genz-Malik rules carry negative weights, so
    +inf in the weighted sum yields inf-inf = NaN. decayV3 therefore returns NaN
    over most of the spectrum once epsrel <~ 1e-4, and decayV2 does the same at
    1e-6. See tests/decay_v2_v3/README.md for the measurements.

    THE SUBSTITUTION. u(v) is affine in v; write u(v) = u0 + beta*v with

        u0 = ma*E_abc/(w_Tq w_Tp),     beta = t*Q*pT/(w_Tq w_Tp),

    so u0 = u(v=0) and u(v1) = 1 defines v1 = (1-u0)/beta, and factor

        u^2 - 1 = beta (v - v1)(u + 1),     1 - v^2 = (1 - v)(1 + v).

    CAUTION ON NAMES: `beta` above is NOT the branching ratio B of
    decayformula.tex. That B is a constant (the --B option) and is applied once
    at the very end, in decayspec(); beta depends on t and pT and lives only
    inside the integrand. They are unrelated.

    With a = max(v1,-1) the lower limit, b = 1 the upper, and

        v(theta) = (a+1)/2 + (1-a)/2 * sin(theta),   theta in [-pi/2, pi/2],

    we have dv = (1-a)/2 * cos(theta) dtheta and (v-a)(1-v) = ((1-a)/2)^2
    cos^2(theta), so the cos(theta) of the Jacobian cancels the square root of
    the two endpoint factors EXACTLY:

        dv * g(v) = C(t) * sqrt( (v-a) / ( beta (v-v1)(u+1)(1+v) ) ) dtheta,
        C(t) = t * primespec(t*Q) / (w_Tq w_Tp).

    That ratio is evaluated in closed form per case rather than as a numerical
    0/0:

        a == v1  (v1 > -1) :  C / sqrt( beta (u+1)(1+v) )
        a == -1  (v1 <= -1):  C / sqrt( (u-1)(u+1) )

    Both are bounded, so nothing diverges at the sampled faces and Cuhre
    integrates a smooth function.

    ONE RESIDUAL, INHERITED FROM THE FORMULA. The substitution removes a
    SQUARE-ROOT endpoint. It cannot remove the case v1 == -1, where the two
    lower branch points collide: (v-v1) and (1+v) then vanish together and the
    integrand behaves like C/(2 sqrt(beta) (v+1)), a 1/(v+1) pole, so the inner
    integral is logarithmically divergent at that single t. This happens only
    for pT < p_abc (for pT >= p_abc one has min_t v1(t) >= 0, well clear of -1),
    it belongs to decayformula.tex rather than to any implementation - V2 and V3
    inherit it identically - and the physical answer stays finite because
    int dt log(1/|t-t*|) converges. In practice V4 handles the region without
    NaN; its self-convergence there degrades to ~7e-4 against ~1e-5 elsewhere.
    So the bound above is uniform in t only for pT >= p_abc.

    A useful consistency check on the algebra: as t -> tmin or tmax the
    v-interval shrinks to a point, yet the inner integral does NOT vanish, since
    int_{v1}^{1} dv/sqrt((v-v1)(1-v)) = pi independently of the width. The
    transformed integrand reproduces this automatically - it has no leftover
    factor of (1-a) - whereas the untransformed version only gets there by
    resolving a singularity against a vanishing Jacobian.

    Everything else (prefactor, bounds tmin/tmax, CLI, output format) is
    identical to decayV3, so the outputs are directly comparable.
*/

/// DEFINE HELPER FUNCTIONS
double w(double p, double m) { return sqrt(p*p + m*m);};

/// u*(t,v) of the reference, kept for reference/debugging; the integrand below
///  uses the equivalent affine form u = u0 + beta*v.
double u_star(double t, double v, double p, double ma, double mb, double E_abc, double Q){
    return (ma * E_abc + t*Q*p*v)/(w(Q*t,ma) * w(p,mb));
};

/// THE SINE-SUBSTITUTED INTEGRAND, UP TO A FACTOR of B*Q*Q*ma/(p_abc * M_PI);
///  ARGUMENTS ARE t AND theta (NOT v). RETURNS THE theta-DENSITY, I.E. THE
///  dv->dtheta JACOBIAN IS ALREADY INCLUDED.
double myintegrand(std::function<double(double)>* primespec, double t, double theta,
                   double p, double ma, double mb, double E_abc, double Q)
{
    const double wq = w(Q*t, ma), wp = w(p, mb);
    const double u0   = ma * E_abc / (wq * wp);   /// u(v) = u0 + beta*v
    const double beta = t * Q * p  / (wq * wp);   /// NOT the branching ratio

    /// u(v=1) = u0 + beta is the largest u on the strip; if it does not exceed 1
    ///  the heavyside kills the whole inner integral at this t.
    if (u0 + beta <= 1.0) return 0.0;

    /// v1 > -1  <=>  u(v=-1) = u0 - beta < 1. The boundary case u0-beta == 1
    ///  is sent to the v1 branch, where the expression stays finite.
    /// At pT == 0 we have beta == 0 (u is then v-independent) and this lands in
    ///  the a == -1 branch, avoiding the 0/0 in (1-u0)/beta.
    const bool lowerIsV1 = (u0 - beta <= 1.0);
    const double a = lowerIsV1 ? (1.0 - u0) / beta : -1.0;

    const double v = 0.5*(a + 1.0) + 0.5*(1.0 - a) * sin(theta);
    const double u = u0 + beta * v;

    const double C = t * (*primespec)(t*Q) / (wq * wp);

    const double val = lowerIsV1 ? C / sqrt(beta * (u + 1.0) * (1.0 + v))
                                 : C / sqrt((u - 1.0) * (u + 1.0));

    /// The two branch points of the integrand collide when v1 == -1 exactly, and
    ///  there the inner integral is genuinely log-divergent - a pre-existing
    ///  feature of the formula, not of this substitution, reachable only for
    ///  pT < p_abc and only on a measure-zero set of t. Returning 0 there keeps
    ///  a single point from poisoning the whole cubature with NaN.
    return std::isfinite(val) ? val : 0.0;
};

double tmin(double ma, double mb, double E_abc, double p_abc, double pT, double Q)    {
    double tmin_ = ma * (E_abc*pT - w(pT,mb) * p_abc)/(Q*mb*mb);
    tmin_ = tmin_ > 0 ? tmin_ : 0;
    return tmin_;
}

double tmax(double ma, double mb, double E_abc, double p_abc, double pT, double Q, double qmax){
    double tmax_ = ma * (E_abc*pT + w(pT,mb) * p_abc)/(Q*mb*mb);
    return tmax_ < qmax/Q ? tmax_ : qmax/Q;
}

/// PARAMETERS FOR THE CUBA INTEGRAND. SELF-CONTAINED HERE SO THAT
///  include/integrate_nested.h - USED BY decayV3 - IS LEFT UNTOUCHED.
struct argsV4 {
    double p, tmin, tmax;
    std::function<double(double)>* primespec;
    double ma, mb, E_abc, Q;
};

/// CUBA INTEGRATES OVER [0,1]^2; MAP IT ONTO (t,theta) ON
///  [tmin,tmax] x [-pi/2,pi/2]. THE DOMAIN IS A PLAIN RECTANGLE AFTER THE
///  SUBSTITUTION - THE v-DEPENDENCE OF THE LOWER LIMIT HAS BEEN ABSORBED - SO NO
///  x1-DEPENDENT INNER BOUNDS AND NO VANISHING JACOBIAN.
static int integrandCUBA(
    const int *ndim, const double xx[],
    const int *ncomp, double ff[], void *userdata)
{
    argsV4* a = (struct argsV4*)userdata;

    const double t     = a->tmin + (a->tmax - a->tmin) * xx[0];
    const double theta = -M_PI_2 + M_PI * xx[1];
    const double jac   = (a->tmax - a->tmin) * M_PI;

    ff[0] = jac * myintegrand(a->primespec, t, theta, a->p, a->ma, a->mb, a->E_abc, a->Q);

    return 0;
}

std::vector<double> decayspec(
    std::vector<double> pTs,
    double ma,
    double mb,
    double mc,
    double B,
    std::function<double(double)> primespec,
    double qmax,
    double Q,
    double epsabs,
    double epsrel,
    int iterations,
    std::function<void(double, double)> callback= [](double p, double val){ return; })
{
    double  p_abc = 1/(2*ma) * sqrt( (pow(ma+mb,2)-pow(mc,2)) * (pow(ma-mb,2)-pow(mc,2)) ),
            E_abc = sqrt(mb*mb + p_abc*p_abc);
    std::vector<double> finalspec(pTs.size());

    argsV4 myargs;
    myargs.ma = ma;
    myargs.mb = mb;
    myargs.E_abc = E_abc;
    myargs.Q = Q;
    myargs.primespec = &primespec;

    const int   NDIM(2),
                NCOMP(1),
                NVEC(1),
                FLAGS(0),
                MINEVAL(0),
                MAXEVAL(iterations),
                KEY(0);
    void* USERDATA(&myargs);
    void* SPIN(NULL);
    const double EPSREL(epsrel), EPSABS(epsabs);
    const char* STATEFILE(NULL);

    int nregions, neval, fail;
    double integral[NCOMP], error[NCOMP], prob[NCOMP];

    int nfail(0), nnonfinite(0);
    for (int i = 0; i < pTs.size(); i++)
    {
        std::cout << i + 1 << " / " << pTs.size() << std::endl;

        double t0 = tmin(ma, mb, E_abc, p_abc, pTs[i], Q);
        double t1 = tmax(ma, mb, E_abc, p_abc, pTs[i], Q, qmax);

        /// The physical support can lie entirely beyond the tabulated primary
        ///  spectrum (tmin > qmax/Q); the truncated integral is then empty.
        if (!(t1 > t0)) {
            finalspec[i] = 0.0;
            callback(pTs[i], finalspec[i]);
            continue;
        }

        myargs.p = pTs[i];
        myargs.tmin = t0;
        myargs.tmax = t1;

        Cuhre(NDIM, NCOMP, integrandCUBA, USERDATA, NVEC,
            EPSREL, EPSABS, FLAGS,
            MINEVAL, MAXEVAL, KEY,
            STATEFILE, SPIN,
            &nregions, &neval, &fail, integral, error, prob);

        /// decayV2/V3 ignore both of these. Cuhre sets fail != 0 when it could
        ///  not reach the requested accuracy within MAXEVAL, and a non-finite
        ///  result must never be reported as a number.
        if (fail != 0) {
            nfail++;
            DEBUGMSG("Cuhre did not reach the requested accuracy at pT = " << pTs[i]
                     << " (fail = " << fail << ", neval = " << neval
                     << ", est.rel.err = " << (integral[0] != 0 ? error[0]/std::abs(integral[0]) : 0)
                     << ")");
        }
        if (!std::isfinite(integral[0])) {
            nnonfinite++;
            DEBUGMSG("non-finite result at pT = " << pTs[i] << " - reporting 0");
            integral[0] = 0.0;
        }

        finalspec[i] = B * integral[0] * Q*Q*ma / (p_abc * M_PI);

        callback(pTs[i],finalspec[i]);
    }

    if (nfail)       DEBUGMSG(nfail << " / " << pTs.size()
                              << " points did not reach the requested accuracy");
    if (nnonfinite)  DEBUGMSG(nnonfinite << " / " << pTs.size()
                              << " points came back non-finite");
    if (!nfail && !nnonfinite)
        DEBUGMSG("all " << pTs.size() << " points converged to the requested accuracy");

    return finalspec;
}

int main(int ac, char* av[])
{
    /// SET UP DEFAULT DIRECTORY NAME TO SAVE TO
    std::time_t t = std::time(nullptr);
    std::tm tm = *std::localtime(&t);
    std::stringstream timestamp_sstr;
    timestamp_sstr << std::put_time(&tm, "%Y%m%d_%H%M%S");
    std::string timestamp = timestamp_sstr.str();

    /* #region COMMAND LINE OPTIONS */
    /// DECLARE SUPPORTED OPTIONS
    namespace po = boost::program_options;
    po::options_description desc(   "// ============================================================= \\\\\n"
                                    "|| This program computes the induced pT-spectrum of a particle b ||\n"
                                    "|| from the decay of a primary particle a via the decay process  ||\n"
                                    "||                       a -> b +c                               ||\n"
                                    "||                                                               ||\n"
                                    "|| Same map as decayV3, with the inner integration variable      ||\n"
                                    "|| sine-substituted so both endpoint singularities cancel        ||\n"
                                    "|| analytically. decayV3 returns NaN for epsrel <~ 1e-4.         ||\n"
                                    "\\\\ ============================================================= //\n\n"
                                    "Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("ma", po::value<double>()->default_value(0.8), "mass of particle a")
        ("mb", po::value<double>()->default_value(0.14), "mass of particle b")
        ("mc", po::value<double>()->default_value(0.14), "mass of particle c")
        ("pTmax", po::value<double>()->default_value(2.0), "spectrum is computed on [0,pTmax]")
        ("NpT", po::value<int>()->default_value(200), "number of sample points within [0,pTmax]")
        ("epsabs",          po::value<double>()->default_value(0),      "absolute integration error goal")
        ("epsrel",          po::value<double>()->default_value(1e-3),   "relative integration error goal")
        ("iter",            po::value<int>()->default_value(1e4),       "maximum integration iterations")
        ("primespecpath",po::value<std::string>(),"csv file containing primary spectrum")
        ("parentdir",       po::value<std::string>()->default_value("Data"),
            "data folder, in which a subfolder for the results is created")
         ("foldername",      po::value<std::string>()->default_value("decay_"+timestamp),
            "target folder for this computation. default is a timestamp 'decay_YYmmdd_HHMMSS', this might be insufficient "
            "if files are created within a second")
        ("B", po::value<double>()->default_value(1.0), "branching ratio of the decay")
        ("Q", po::value<double>()->default_value(1.0), "dimensionful pT-scale (should not influence the result)")
        ;

    po::variables_map vm;
    po::store(po::parse_command_line(ac, av, desc), vm);
    po::notify(vm);

    /// PROCESS CMDLINE OPTIONS
    if (vm.count("help"))    {
        std::cout << desc << "\n";
        return 1;
    }

    DEBUGMSG("start command line processing");
    double  ma                  = vm["ma"].as<double>(),
            mb                  = vm["mb"].as<double>(),
            mc                  = vm["mc"].as<double>(),
            pTmax               = vm["pTmax"].as<double>(),
            B                   = vm["B"].as<double>(),
            Q                   = vm["Q"].as<double>(),
            epsabs              = vm["epsabs"].as<double>(),
            epsrel              = vm["epsrel"].as<double>();
    int NpTs                    = vm["NpT"].as<int>(),
        iter                    = vm["iter"].as<int>();
    std::string primespecpath   = vm["primespecpath"].as<std::string>(),
                parentdir       = vm["parentdir"].as<std::string>(),
                foldername      = vm["foldername"].as<std::string>();

    /// The Kallen factor turns negative when the decay is kinematically closed;
    ///  decayV2/V3 would silently produce NaN from here on.
    double kallen = (pow(ma+mb,2)-pow(mc,2)) * (pow(ma-mb,2)-pow(mc,2));
    if (kallen < 0) {
        std::cerr << "Error: the decay a -> b + c is kinematically forbidden for "
                  << "ma = " << ma << ", mb = " << mb << ", mc = " << mc
                  << " (requires ma >= mb + mc)." << std::endl;
        return -1;
    }
    double  p_abc               = 1/(2*ma) * sqrt(kallen),
            E_abc               = sqrt(mb*mb + p_abc*p_abc);
    DEBUGMSG("command line processing completed");
    /* #endregion */

    /// INTERPOLATE PRIMESPEC FROM DATA
    DEBUGMSG("start data processing");
    csvdata primespecdata = readcsv(primespecpath);
    std::vector<double> x(primespecdata.data[0]);
    std::vector<double> y(primespecdata.data[1]);
    double qmin(x[0]), qmax(x[x.size()-1]);

    /// The t-integration starts at t = 0 whenever pT < p_abc, so the spline is
    ///  asked for q = 0. Report that instead of letting pchip throw.
    if (qmin > 0)
        DEBUGMSG("warning: primary spectrum starts at qT = " << qmin << " > 0; "
                 "the decay integral samples down to qT = 0 and will extrapolate");

    std::vector<double> log10y(y.size()); /// logarithmic data is probably interpolated better
    for(int i = 0; i < y.size(); i++) {
        if (!(y[i] > 0)) {
            std::cerr << "Error: primary spectrum is non-positive at qT = " << x[i]
                      << " (value " << y[i] << "); cannot interpolate log10." << std::endl;
            return -1;
        }
        log10y[i] = log10(y[i]);
    }

    using boost::math::interpolators::pchip;
    auto logspecspline = pchip(
        std::move(x),
        std::move(log10y));
    /// The query is clamped to the tabulated range. For a spectrum produced by
    ///  specV2 this is a no-op (it starts at qT = 0 and the t-integration never
    ///  exceeds qmax/Q), but for an externally supplied table starting above 0
    ///  it turns a pchip out-of-domain throw into constant extrapolation, which
    ///  the warning above has already flagged.
    std::function<double(double)> primespec = [&,qmin,qmax](double p){
        double q = p < qmin ? qmin : (p > qmax ? qmax : p);
        return pow(10, logspecspline(q));
    };
    DEBUGMSG("data processing completed");

    /// CREATE DIRECTORY TO SAVE FILES
    std::string pathname = parentdir+"/"+foldername;
    std::filesystem::create_directories(pathname); /// CREATE DIRECTORY HERE

    int NSAMPLE(1000);
    writeFunctionsToFile<double>(    pathname+"/primespec.txt",
                                    {primespec},
                                    0, qmax, NSAMPLE,
                                    {"qT","primespec"},
                                    {timestamp}
                        );

    /// AGAIN, WRITE TO FILE DURING COMPUTATION, SO PREPARE THE FILE BEFOREHAND
    std::stringstream qmax_ss, primespec_ss, userinput_ss;
    qmax_ss         << "qmax:\t" << qmax;
    primespec_ss    << "primespec:\t" << primespecpath;
    userinput_ss    << "ma:\t" << ma << "\n# "  /// the comments above are just 1 line per comment. Here we have multiple lines, therefore add the comment symbol "#" manually....
                    << "mb:\t" << mb << "\n# "
                    << "mc:\t" << mc << "\n# "
                    << "[=>] pabc:\t" << p_abc << "\n# "
                    << "[=>] Eabc:\t" << E_abc << "\n# "
                    << "B:\t" << B << "\n# "
                    << "Q:\t" << Q << "\n# "
                    << "epsrel:\t" << epsrel << "\n# "
                    << "epsabs:\t" << epsabs << "\n# "
                    << "iter:\t" << iter;
    /// KEEP THIS LAYOUT BYTE-COMPATIBLE WITH decayV2/decayV3: evaluate.ipynb
    ///  reads the parent mass as float(lines[3].replace("# ma:\t","")), i.e. by
    ///  HARD-CODED LINE INDEX. Any comment inserted before userinput_ss shifts
    ///  that index and breaks the notebook, so the decayV4 marker goes LAST.
    std::vector<std::string> comments({
        timestamp,
        primespec_ss.str(),
        qmax_ss.str(),
        userinput_ss.str(),
        "decayV4 (sine-substituted inner integral)",
    });
    std::vector<std::string> headers({"pT","decayspec"});

    std::string path = pathname+"/decayspec.txt";
    std::ofstream decayspec_output(path);
    if (!decayspec_output.is_open()) { std::cerr << "Error opening the file: " << path << " to save to" << std::endl; return -1; }
    for(int i = 0; i < comments.size(); i++) decayspec_output << "# " << comments[i] << std::endl;
    for(int i = 0; i < headers.size(); i++) { decayspec_output << headers[i]; if(i != headers.size()-1) decayspec_output << ","; }
    decayspec_output << std::endl;

    /// COMPUTE DECAY SPEC
    std::vector<double> pTs(NpTs);
    for (int i = 0; i < pTs.size(); i++) pTs[i] = i * pTmax / (NpTs - 1);

    std::function<void(double,double)> callback = [&decayspec_output](double p, double value)
    {
        decayspec_output << p << "," << value << std::endl;
    };

    std::vector<double> finalspec = decayspec(pTs, ma, mb, mc, B, primespec, qmax, Q, epsabs, epsrel, iter, callback);
    decayspec_output.close();
}
