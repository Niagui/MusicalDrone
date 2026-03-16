#include <stdio.h>
#include <stdlib.h>
#include "osqp.h"

//do a <make osqp_test> to check if the osqp is successfully built

int main() 
{
    // 1. Use standard malloc or simply declare on the stack
    OSQPSettings settings; 
    
    // 2. This function confirms the header AND the library are linked
    osqp_set_default_settings(&settings);
    
    printf("OSQP successfully imported!\n");
    printf("Default rho: %f\n", settings.rho);

    return 0;
}