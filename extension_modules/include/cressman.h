#ifndef CRESSMAN_H
#define CRESSMAN_H


#include <cmath>
#include <iostream>


class Cressman
{
    public:
        Cressman(const double Cm, const double GNa, const double GK, const double GAHP,
            const double GKL, const double GNaL, const double GClL, const double GCa, const double Gglia,
            const double Koinf, const double gamma1, const double tau, const double control)
            : Cm(Cm), GNa(GNa), GK(GK), GAHP(GAHP), GKL(GKL), GNaL(GNaL),
              GClL(GClL), GCa(GCa), Gglia(Gglia), Koinf(Koinf),
              gamma1(gamma1), tau(tau), control(control) { }

        template< class vector_type >
        void operator() (const vector_type &x, vector_type &dxdt, const double /* t */ ) {
            const double Ipump = rho*(1/(1 + exp((25 - x[6])/3.)))*(1/(1 + exp(5.5 - x[5])));
            const double IGlia = Gglia/(1 + exp((18.- x[5])/2.5));
            const double Idiff = eps0*(x[5] - Koinf);

            const double Ki = 140 + (18 - x[6]);
            const double Nao = 144 - beta0*(x[6] - 18);
            const double ENa = 26.64*log(Nao/x[6]);
            const double EK = 26.64*log(control*x[5]/Ki);

            const double am = (0.1*x[0] + 3.0)/(1 - exp(-0.1*x[0] - 3));
            const double bm = 4*exp(-1./18.*x[0] - 55./18.);
            const double an = (0.01*x[0] + 0.34)/(-exp(-0.1*x[0] - 3.4) + 1);
            const double bn = 0.125*exp(-0.0125*x[0] - 0.55);
            const double ah = 0.07*exp(-0.05*x[0] - 2.2);
            const double bh = 1.0/(exp(-0.1*x[0] - 1.4) + 1);

            const double taum = 1./(am + bm);
            const double minf = 1./(am + bm)*am;
            const double taun = 1./(an + bn);
            const double ninf = 1./(an + bn)*an;
            const double tauh = 1./(bh + ah);
            const double hinf = 1./(bh + ah)*ah;

            const double INa = GNa*pow(x[1], 3)*x[3]*(x[0] - ENa) + GNaL*(x[0] - ENa);
            const double IK = (GK*pow(x[2], 4) + GAHP*x[4]/(1 + x[4]) + GKL)*(x[0] - EK);
            const double ICl = GClL*(x[0] - ECl);

            dxdt[0] = -(INa + IK + ICl)/Cm;
            dxdt[1] = phi*(minf - x[1])/taum;
            dxdt[2] = phi*(ninf - x[2])/taun;
            dxdt[3] = phi*(hinf - x[3])/tauh;
            dxdt[4] = x[4]/80. - 0.002*GCa*(x[0] - ECa)/(1 + exp(-(x[0] + 25.)/2.5));
            dxdt[5] = (gamma1*beta0*IK - 2*beta0*Ipump - IGlia - Idiff)/tau;
            dxdt[6] = -(gamma1*INa + 3*Ipump)/tau;
        }

    private:
        const double rho = 1.25;
        const double eps0 = 1.2;
        const double beta0 = 7.0;
        const double ECa = 120.0;
        const double Cli = 6.0;
        const double Clo = 130.0;
        const double ECl = 26.64*log(Cli/Clo);
        const double phi = 3.0;

        const double Cm;
        const double GNa;
        const double GK;
        const double GAHP;
        const double GKL;
        const double GNaL;
        const double GClL;
        const double GCa;
        const double Gglia;
        const double Koinf;
        const double gamma1;
        const double tau;
        const double control;
};


#endif
