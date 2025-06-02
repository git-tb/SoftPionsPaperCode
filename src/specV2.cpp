#include <iostream>
#include <cmath>    /// bessel function std::cyl_bessel_j, std::cyl_neumann
                    /// but also j0, j1, y0, y1 via <math.h>
#include <complex>
#include <functional>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>
#include <map>
#include <boost/math/interpolators/pchip.hpp>
#include <boost/program_options.hpp>
#include <filesystem>
#include <ctime>   /// std::time, std::local_time
#include <iomanip> /// std::put_time
#include <omp.h>

#include <algorithm>
#include <iterator>

#include "savedata.h"       /// write to file
#include "freezeout.h"      /// freezeout geometry
#include "initdata.h"       /// initial data for condensate field
#include "constants.h"      /// GeV to inverse fm,...

#include "debugmsg.h"

using namespace std::complex_literals;

/* #region HELPER FUNCTIONS FOR SPECTRUM COMPUTATION */

/// GSL_FUNCTION (NEEDED BY GSL INTEGRATORS) ACCEPTS ONLY AN INTEGRATION VARIABLE (double)
///  AND PARAMETERS (void*), SO WE STORE ALL ADDITIONAL INFORMATION IN A COMPACT OBJECT
struct intargs {
    double pT, m;
    std::function<std::complex<double>(double)> func, Dfunc;
    freezeoutFunctions fo;
    bool anti;

    intargs() = delete; /// NO ACCIDENTAL INITIALIZATION
    intargs(double pT_,
            double m_,
            std::function<std::complex<double>(double)> func_,
            std::function<std::complex<double>(double)> Dfunc_,
            freezeoutFunctions fo_,
            bool anti_) :
                pT(pT_), m(m_), func(func_), Dfunc(Dfunc_), fo(fo_), anti(anti_)
            {}
};

double omega(double p, double m) { return sqrt(p * p + m * m); }
/// double J0(double x) { return std::cyl_bessel_j(0, x); } /// THESE SEEM TO BE UNOPTIMIZED FOR INTEGER ORDER AND ARE SLOWER BY A FACTOR OF ~10 COMPARED TO THE ALTERNATIVE j0,j1,y0,y1
/// double J1(double x) { return std::cyl_bessel_j(1, x); }
/// double Y0(double x) { return std::cyl_neumann(0, x); }
/// double Y1(double x) { return std::cyl_neumann(1, x); }
double J0(double x) { return j0(x); }
double J1(double x) { return j1(x); }
double Y0(double x) { return y0(x); }
double Y1(double x) { return y1(x); }

/// GSL_FUNCTION (NEEDED BY GSL INTEGRATORS) CANNOT BE INITIALIZED WITH STD::FUNCTION OBJECTS,
///  FUNCTION POINTERS SEEM TO WORK, SO WE USE THOSE FUNCTION POINTERS (*integrand)
std::complex<double> (*myintegrand)(double, void*) = [](double alpha, void* params) {
    /*
        returns the full integrand at given alpha with the factor of 2*pi^2
    */

    /*
    THE FOLLOWING WORKS, BUT IS SLOW BECAUSE THE POINTER IS DEREFERENCED (*) AND A COPY OF THE UNDERLYING
    PARAMETER OBJECT - CONTAINING THE INTERPOLATING FUNCTION FROM THE FREEZEOUT SURFACE - IS 
    CREATED AT EVERY FUNCTION CALL. ITS BETTER TO KEEP ONLY THE POINTER.

    WE KEEP THE CODE IN THE COMMENT JUST IN CASE WE NOTICE PROBLEMS WITH THE POINTER USAGE.
    */ 
    /// intargs args = *(struct intargs *)params;
    /// double  m(args.m),
    ///         pT(args.pT),
    ///         wT(omega(pT,m)),
    ///         r(args.fo.r(alpha)       * fmtoIGeV),
    ///         Dr(args.fo.Dr(alpha)     * fmtoIGeV),
    ///         tau(args.fo.tau(alpha)   * fmtoIGeV),
    ///         Dtau(args.fo.Dtau(alpha) * fmtoIGeV);

    /// double pm = (args.anti ? -1.0 : +1.0);
    /// return 2 * M_PI * M_PI * r * tau * (
    ///     args.Dfunc(alpha) * (
    ///                     J0(r*pT) * ( -Y0(tau*wT) + pm * 1i * J0(tau*wT))
    ///     ) +
    ///     args.func(alpha) * (
    ///         Dtau*pT *   J1(r*pT) * ( -Y0(tau*wT) + pm * 1i * J0(tau*wT)) +
    ///         Dr*wT *     J0(r*pT) * ( -Y1(tau*wT) + pm * 1i * J1(tau*wT))
    ///     )
    /// );

    /// AGAIN, THIS TIME WITH THE APPROPRIATE POINTER USAGE
    intargs* args = (struct intargs *)params;
    double  m(args->m),
            pT(args->pT),
            wT(omega(pT,m)),
            r(args->fo.r(alpha)       * fmtoIGeV),
            Dr(args->fo.Dr(alpha)     * fmtoIGeV),
            tau(args->fo.tau(alpha)   * fmtoIGeV),
            Dtau(args->fo.Dtau(alpha) * fmtoIGeV);

    double pm = (args->anti ? -1.0 : +1.0);
    return 2 * M_PI * M_PI * r * tau * (
        args->Dfunc(alpha) * (
                        J0(r*pT) * ( -Y0(tau*wT) + pm * 1i * J0(tau*wT))
        ) +
        args->func(alpha) * (
            Dtau*pT *   J1(r*pT) * ( -Y0(tau*wT) + pm * 1i * J0(tau*wT)) +
            Dr*wT *     J0(r*pT) * ( -Y1(tau*wT) + pm * 1i * J1(tau*wT))
        )
    );
};
/* #endregion */

/* #region SPECTRUM COMPUTATION FOR LIST OF PT-VALUES */
std::vector<std::complex<double>> spectr_Jp(
    std::vector<double> pTs,
    double m,
    std::function<std::complex<double>(double)> func,
    std::function<std::complex<double>(double)> Dfunc,
    freezeoutFunctions fo,
    double epsabs,
    double epsrel,
    int iterations,
    bool anti = false,
    std::function<void(double, std::complex<double>)> callback = [](double p, std::complex<double> val){ return; })
{
    std::vector<std::complex<double>> result(pTs.size());

    int key(6); /// 61 point Gauss-Kronrod rule https://www.gnu.org/software/gsl/doc/html/integration.html
    gsl_set_error_handler_off();

    /// IF EXECUTED IN PARALLEL, EACH THREAD CREATES ITS OWN WORKSPACE
    /// IF EXECUTED IN SERIES, SAVE TIME BY REUSING THE WORKSPACE
    #if not defined(_OPENMP)
    gsl_integration_workspace *workspace = gsl_integration_workspace_alloc(iterations);
    #endif

    int totalexecuted(0);
    #pragma omp parallel for default(none) shared(totalexecuted, callback, fo, m, result, anti, std::cout, pTs, func, Dfunc, myintegrand, epsabs, epsrel, iterations, key)
    for (int i = 0; i < pTs.size(); i++)
    {
        #pragma omp critical
        std::cout << i << " / " << pTs.size() << " (" << 100 * (double)totalexecuted/(double)pTs.size() << "%)" << std::endl;
        totalexecuted++;

        #if defined(_OPENMP)
        gsl_integration_workspace *workspace = gsl_integration_workspace_alloc(iterations);
        #endif

        double pT = pTs[i];
        intargs myargs(pT, m, func, Dfunc, fo, anti);
        gsl_function F_re, F_im;
        F_re.function = [](double alpha, void* params){ return std::real(myintegrand(alpha, params)); };
        F_im.function = [](double alpha, void* params){ return std::imag(myintegrand(alpha, params)); };
        F_re.params = &myargs;
        F_im.params = &myargs;

        double result_re, result_im, error_re, error_im;

        int status = gsl_integration_qag(&F_re, 0, M_PI_2, epsabs, epsrel, iterations, key, workspace, &result_re, &error_re);
        if (status)
            #pragma omp critical
            DEBUGMSG(gsl_strerror(status) << " at p = " << myargs.pT << " | estimated error (re): " << error_re);

        status = gsl_integration_qag(&F_im, 0, M_PI_2, epsabs, epsrel, iterations, key, workspace, &result_im, &error_im);
        if (status)
            #pragma omp critical
            DEBUGMSG(gsl_strerror(status) << " at p = " << myargs.pT << " | estimated error (imag): " << error_im);

        result[i] = result_re + 1i * result_im;

        /// IN PARALLEL PROCEDURE, THIS HAS NO EFFECT
        callback(pTs[i],result[i]);
    
        #if defined(_OPENMP)
        gsl_integration_workspace_free(workspace);
        #endif
    }
    
    #if not defined(_OPENMP)
    gsl_integration_workspace_free(workspace);
    #endif

    return result;
}
/* #endregion */

int main(int ac, char* av[])
{

    /// SET UP DEFAULT DIRECTORY NAME TO SAVE TO
    std::time_t t = std::time(nullptr);
    std::tm tm = *std::localtime(&t);
    std::stringstream timestamp_sstr;
    timestamp_sstr << std::put_time(&tm, "%Y%m%d_%H%M%S");
    std::string timestamp = timestamp_sstr.str();

    /// DECLARE SUPPORTED COMMAND LINE OPTIONS
    namespace po = boost::program_options;
    po::options_description desc(   "//================================================================= \\\\\n"
                                    "|| This program computes the pT-spectrum of particles associated    ||\n"
                                    "|| to a condensate field during a heavy-ion collision. The field    ||\n"
                                    "|| is sourced by some unkown Bjorken-(tau,r)-dependent interaction, ||\n"
                                    "|| but is supposed to be known on the freezeout surface in the      ||\n"
                                    "|| context of a hydrodynamic evolution of the QGP.                  ||\n"
                                    "\\\\ ================================================================ //\n\n"
                                    "Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("initpath",        po::value<std::string>(),                   "csv file containing initial field data")
        ("freezeoutpath",   po::value<std::string>()->default_value("./../../Mathematica/data/ExampleFreezeOutCorrected.csv"),
            "csv file containing freezeout geometry")
        ("m",               po::value<double>()->default_value(0.14),   "particle mass (in GeV)")
        ("pTmax",           po::value<double>()->default_value(1.0),    "spectrum is computed on [0,pTmax] (in GeV)")
        ("NpT",             po::value<int>()->default_value(100),       "number of sample points within [0,pTmax] (in GeV)")
        ("epsabs",          po::value<double>()->default_value(0),      "absolute integration error goal")
        ("epsrel",          po::value<double>()->default_value(1e-3),   "relative integration error goal")
        ("iter",            po::value<int>()->default_value(1e4),       "maximum integration iterations")
        ("parentdir",       po::value<std::string>()->default_value("Data"),
            "data folder, in which a subfolder for the results is created")
        ("foldername",      po::value<std::string>()->default_value("spec_"+timestamp),
            "target folder for this computation. default is a timestamp 'spec_YYmmdd_HHMMSS', this might be insufficient "
            "if files are created within a second")
        ("savegeometry",    po::bool_switch()->default_value(false),    "flag only for debugging, save the freezeout geometry");

    po::variables_map vm;
    po::store(po::parse_command_line(ac, av, desc), vm);
    po::notify(vm);

    /// PROCESS CMDLINE OPTIONS
    if (vm.count("help"))    {
        std::cout << desc << "\n";
        #ifdef _OPENMP
        DEBUGMSG("Parallelization with OPENMP enabled.");
        #else
        DEBUGMSG("Parallelization with OPENMP disabled.");
        #endif
        return 1;
    }

    DEBUGMSG("start command line processing");
    double  pTmin               = 0,
            pTmax               = vm["pTmax"].as<double>(),
            mParticle           = vm["m"].as<double>(),
            epsabs              = vm["epsabs"].as<double>(),
            epsrel              = vm["epsrel"].as<double>();
    int NpT                     = vm["NpT"].as<int>(),
        iter                    = vm["iter"].as<int>();
    std::string initpath        = vm["initpath"].as<std::string>(),
                freezeoutpath   = vm["freezeoutpath"].as<std::string>(),
                parentdir       = vm["parentdir"].as<std::string>(),
                foldername      = vm["foldername"].as<std::string>();
    bool savegeometry           = vm["savegeometry"].as<bool>();
    DEBUGMSG("command line processing completed");

    DEBUGMSG("start data processing");
    freezeoutData foDat = ProcessFreezeoutData(freezeoutpath);
    initData initDat    = ProcessInitialData(initpath);
    DEBUGMSG("data processing completed");

    /// CREATE DIRECTORY TO SAVE FILES
    std::string pathname = parentdir+"/"+foldername;
    std::filesystem::create_directories(pathname);

    /// DEFINE CONDENSATE FIELD AND DERIVATIVE ON FREEZOUT SURFACE
    std::function<std::complex<double>(double)> func = [&initDat](double alpha)
    { return initDat.initFunc.f0Re(alpha) + 1i * initDat.initFunc.f0Im(alpha); };
    std::function<std::complex<double>(double)> Dfunc = [&initDat](double alpha)
    { return initDat.initFunc.Df0Re(alpha) + 1i * initDat.initFunc.Df0Im(alpha); };

    /// SAVE INTERPOLATED FUNCTIONS SAMPLED AT SOME PRESCRIBED RESOLUTION
    int NSAMPLE = 1000;
    writeFunctionsToFile<std::complex<double>>(     pathname+"/initialfields.txt",
                                                    {func, Dfunc},
                                                    0, M_PI / 2.0, NSAMPLE,
                                                    {"alpha","phi","Dphi"},
                                                    {timestamp}
                        );
    if(savegeometry) { 
        /// ONLY IF ONE WANTS TO ENSURE THAT DATA PROCESSING HAS WORKED PROPERLY, 
        ///  THE GEOMETRY DATA IS ALSO SAVED
        writeFunctionsToFile<double>(   pathname+"/geometry.txt",
                                        {foDat.foFunc.tau, foDat.foFunc.r, foDat.foFunc.Dtau, foDat.foFunc.Dr},
                                        0, M_PI / 2.0, NSAMPLE,
                                        {"alpha","tau","r","Dtau","Dr"},
                                        {timestamp}
                            );
    }  

    /// WRITE TO FILE DURING COMPUTATION, THEREFORE PREPARE THE FILE HERE
    std::stringstream initdata_ss, geometry_ss, NpT_ss, pTmax_ss, epsabs_ss, epsrel_ss, iter_ss, mass_ss;
    initdata_ss << "initdata:\t"        << initpath;
    geometry_ss << "freezeout file: \t" << freezeoutpath;
    NpT_ss      << "NpT:\t"             << NpT;
    pTmax_ss    << "pTmax:\t"           << pTmax;
    epsabs_ss   << "epsabs:\t"          << epsabs;
    epsrel_ss   << "epsrel:\t"          << epsrel;
    iter_ss     << "integr iter:\t"     << iter;
    mass_ss     << "particle mass:\t"   << mParticle;
    std::vector<std::string> comments({   
        timestamp,
        initdata_ss.str(),
        geometry_ss.str(),
        mass_ss.str(),
        NpT_ss.str(),
        pTmax_ss.str(),
        epsabs_ss.str(),
        epsrel_ss.str(),
        iter_ss.str()
    });
    std::vector<std::string> headers({"pT","spec"});

    /// COMPUTE SPECTRUM
    std::vector<double> pTs(NpT);
    for (int i = 0; i < pTs.size(); i++)
        pTs[i] = pTmin + i * (pTmax - pTmin) / (NpT - 1);

    /// =======================================
    /// START WITH THE PARTICLE SPECTRUM
    std::string spectr_path = pathname+"/spectr.txt";
    std::ofstream spectr_output(spectr_path);
    if (!spectr_output.is_open()) { std::cerr << "Error opening the file: " << spectr_path << " to save to" << std::endl; return -1; }
    for(int i = 0; i < comments.size(); i++) {
        spectr_output << "# " << comments[i] << std::endl;
    }
    for(int i = 0; i < headers.size(); i++) {
        spectr_output << headers[i]; 
        if(i != headers.size()-1) spectr_output << ",";
    }
    spectr_output << std::endl;

    /// IF THE SPECTRUM COMPUTATION WOULD HAPPEN IN A SERIAL WAY, IT WOULD BE USEFUL TO INTRODUCE A CALLBACK
    ///  FUNCTION THAT SAVES THE DATAPOINTS (p[i],spec(p[i])) DIRECTLY AFTER THE iTH STEP, SUCH THAT THE PROGRAM
    ///  CAN BE INTERRUPTED (OR CRASH) AT ANY POINT AND NOT ALL DATA IS LOST.
    /// IF WE COMPUTE THE VALUES (p[i],spec(p[i])) IN PARALLEL, THE RESULTING DATASET WOULD BE WRITTEN TO FILE
    ///  IN RANDOM ORDER. INSTEAD, WRITE TO AN ORDERED LIST IN THE PARALLEL PROCEDURE AND WRITE TO FILE AFTER
    ///  ALL VALUES ARE COMPUTED. DATA IS LOST WHEN THE PROGRAM CRASHES IN BETWEEN, BUT THE COMPUTATION IS MUCH
    ///  FASTER ANYWAYS, SO WE DONT CARE.
    std::function<void(double,std::complex<double>)> spectr_callback = [&spectr_output](double p, std::complex<double> value) {
        #if not defined(_OPENMP)
        double abs2norm = 0.5 * (1 / std::pow(2 * M_PI, 3)) * std::norm(value);
        spectr_output << p << "," << std::real(abs2norm) << std::endl;
        #endif
    };   

    /// CENTRAL COMPUTATION (PARTICLE SPECTRUM) HAPPENS HERE
    ///  THIS COMPUTES J(p). THE SPECTRUM IS GIVEN BY (1/2)*(1/(2PI)^3)*|J(p)|^2.
    std::vector<std::complex<double>> myspectr = spectr_Jp( pTs,
                                                            mParticle,
                                                            func,
                                                            Dfunc,
                                                            foDat.foFunc,
                                                            epsabs,
                                                            epsrel,
                                                            iter,
                                                            /*anti=*/false,
                                                            spectr_callback);

    #ifdef _OPENMP
    for (int i = 0; i < pTs.size(); i++) {
        double abs2norm = 0.5 * (1 / std::pow(2 * M_PI, 3)) * std::norm(myspectr[i]);
        spectr_output << pTs[i] << "," << std::real(abs2norm) << std::endl;
    }    
    #endif
    spectr_output.close();

    /// =======================================
    /// ...AND THEN THE ANTI PARTICLE SPECTRUM
    std::string spectranti_path = pathname+"/spectr_anti.txt";
    std::ofstream spectranti_output(spectranti_path);
    if (!spectranti_output.is_open()) { std::cerr << "Error opening the file: " << spectranti_path << " to save to" << std::endl; return -1; }
    for(int i = 0; i < comments.size(); i++) {
        spectranti_output << "# " << comments[i] << std::endl;
    }
    for(int i = 0; i < headers.size(); i++) {
        spectranti_output << headers[i];
        if(i != headers.size()-1) spectranti_output << ",";
    }
    spectranti_output << std::endl;

    /// ...
    /// (((SAME COMMENT ABOUT SERIAL AND PARALLEL EXECUTION...)))
    /// ...
    std::function<void(double,std::complex<double>)> spectranti_callback = [&spectranti_output](double p, std::complex<double> value) {
        #if not defined(_OPENMP)
        double abs2norm = 0.5 * (1 / std::pow(2 * M_PI, 3)) * std::norm(value);
        spectranti_output << p << "," << std::real(abs2norm) << "," << std::imag(abs2norm) << std::endl;
        #endif
    };

    /// CENTRAL COMPUTATION (ANTIPARTICLE SPECTRUM) HAPPENS HERE
    ///  THIS COMPUTES J(-p). THE SPECTRUM IS GIVEN BY (1/2)*(1/(2PI)^3)*|J(-p)|^2.
    std::vector<std::complex<double>> myspectr_anti = spectr_Jp(pTs,
                                                                mParticle,
                                                                func,
                                                                Dfunc,
                                                                foDat.foFunc,
                                                                epsabs,
                                                                epsrel,
                                                                iter,
                                                                /*anti=*/true,
                                                                spectranti_callback);

    #ifdef _OPENMP
    for (int i = 0; i < pTs.size(); i++)
    {
        double abs2norm = 0.5 * (1 / std::pow(2 * M_PI, 3)) * std::norm(myspectr_anti[i]);
        spectranti_output << pTs[i] << "," << std::real(abs2norm) << std::endl;
    }
    #endif
    spectranti_output.close();

    return 0; /// :)
}