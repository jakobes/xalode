#ifndef CRESSMAN_H
#define CRESSMAN_H


#include <vector>
#include <cmath>
#include <iostream>


template< typename vector_type >
class Cressman_step
{
    public:
        Cressman_step(const double Cm, const double GNa, const double GK, const double GAHP,
                const double GKL, const double GNaL, const double GClL, const double GCa,
                const double Gglia, const double Koinf, const double gamma1,
                const double tau, const double control) : Cm(Cm), GNa(GNa), GK(GK),
                GAHP(GAHP), GKL(GKL), GNaL(GNaL), GClL(GClL), GCa(GCa), Gglia(Gglia),
                Koinf(Koinf), gamma1(gamma1), tau(tau), control(control) { }

        void step(const vector_type &V, const vector_type &m, const vector_type &n, const vector_type &h,
                const vector_type &Ca, const vector_type &K, const vector_type &Na, vector_type &V_new,
                vector_type &m_new, vector_type &n_new, vector_type &h_new, vector_type &Ca_new, vector_type &K_new,
                vector_type &Na_new, double dt)
        {
            for (size_t i = 0; i < V.size(); ++i)
            {
                const double Ipump = rho*(1/(1 + exp((25 - Na[i])/3.)))*(1/(1 + exp(5.5- K[i])));
                const double IGlia = Gglia/(1 + exp((18.- K[i])/2.5));
                const double Idiff = eps0*(K[i] - Koinf);

                const double Ki = 140 + (18 - Na[i]);
                const double Nao = 144 - beta0*(Na[i] - 18);
                const double ENa = 26.64*log(Nao/Na[i]);
                const double EK = 26.64*log(control*K[i]/Ki);
                const double ECl = 26.64*log(Cli/Clo);

                const double am = (0.1*V[i] + 3.0)/(-exp(-0.1*V[i] - 3) + 1);
                const double bm = 4*exp(-1./18.*V[i] - 55./18.);
                const double ah = 0.07*exp(-0.05*V[i] - 2.2);
                const double bh = 1.0/(exp(-0.1*V[i] - 1.4) + 1);
                const double an = (0.01*V[i] + 0.34)/(-exp(-0.1*V[i] - 3.4) + 1);
                const double bn = 0.125*exp(-0.0125*V[i] - 0.55);

                const double taum = 1./(am + bm);
                const double minf = 1./(am + bm)*am;
                const double taun = 1./(an + bn);
                const double ninf = 1./(an + bn)*an;
                const double tauh = 1./(bh + ah);
                const double hinf = 1./(bh + ah)*ah;

                const double INa = GNa*pow(m[i], 3)*h[i]*(V[i] - ENa) + GNaL*(V[i] - ENa);
                const double IK = (GK*pow(n[i], 4) + GAHP*Ca[i]/(1 + Ca[i]) + GKL)*(V[i] - EK);
                const double ICl = GClL*(V[i] - ECl);

                V_new[i] = V[i] + dt*(-(INa + IK + ICl)/Cm);
                m_new[i] = m[i] + dt*phi*(minf - m[i])/taum;
                n_new[i] = n[i] + dt*phi*(ninf - n[i])/taun;
                h_new[i] = h[i] + dt*phi*(hinf - h[i])/tauh;
                Ca_new[i] = Ca[i] + dt*(Ca[i]/80. - 0.002*GCa*(V[i] - ECa)/(1 + exp(-(V[i] + 25.)/2.5)));
                K_new[i] = K[i] + dt*((gamma1*beta0*IK - 2*beta0*Ipump - IGlia - Idiff)/tau);
                Na_new[i] = Na[i] + dt*(-(gamma1*INa + 3*Ipump)/tau);
            }
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
            const double Ipump = rho*(1/(1 + exp((25 - x[6])/3.)))*(1/(1 + exp(5.5- x[5])));
            const double IGlia = Gglia/(1 + exp((18.- x[5])/2.5));
            const double Idiff = eps0*(x[5] - Koinf);

            const double Ki = 140 + (18 - x[6]);
            const double Nao = 144 - beta0*(x[6] - 18);
            const double ENa = 26.64*log(Nao/x[6]);
            const double EK = 26.64*log(control*x[5]/Ki);
            const double ECl = 26.64*log(Cli/Clo);

            const double am = (0.1*x[0] + 3.0)/(-exp(-0.1*x[0] - 3) + 1);
            const double bm = 4*exp(-1./18.*x[0] - 55./18.);
            const double ah = 0.07*exp(-0.05*x[0] - 2.2);
            const double bh = 1.0/(exp(-0.1*x[0] - 1.4) + 1);
            const double an = (0.01*x[0] + 0.34)/(-exp(-0.1*x[0] - 3.4) + 1);
            const double bn = 0.125*exp(-0.0125*x[0] - 0.55);

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
