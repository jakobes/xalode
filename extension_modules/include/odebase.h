#ifndef ODEBASE_H
#define ODEBASE_H

// stl
#include <memory>     // Enable_shared_from_this
#include <vector>
#include <iostream>
#include <string>


/* class ODEBase: public std::enable_shared_from_this< ODEBase > */
class ODEBase
{
    public:
        typedef std::vector< double > vector_type;

        virtual std::shared_ptr< ODEBase > clone() const
        {
            return std::make_shared< ODEBase >(*this);
        }

        virtual void operator() (const vector_type &x, vector_type &dxdt, const double /* t */) const { }

        virtual void eval(const vector_type &x, vector_type &dxdt, const double /* t */) const { }

        virtual void print() const
        {
            std::cout << "Foo" << std::endl;
        }
};


#endif
