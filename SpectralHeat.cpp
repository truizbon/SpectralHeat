#include <iostream>
#include <cmath>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <chrono>


#define PI 3.14159265359

std::vector<double> initial_distribution(std::vector<double> x);
double lagrange_basis_derivative(std::vector<double> quad_points, int n, double x);
double lagrange_basis_polynomial(std::vector<double> quad_points, int count_i, double x);
std::vector<std::vector<double> > lglnodes(int n);
std::vector<std::vector<double> > spectral_heat(int p, int num_elem, double l, double nu, double final_time);
std::vector<std::vector<double> > inverse(std::vector<std::vector<double> > m);

int main(int argc, char *argv[]) {

    if (argc != 6) {
        std::cout << "Usage: " << argv[0] << " p num_elem l nu final_time" << std::endl;
        return 1;
    }

    int p = atoi(argv[1]);
    int num_elem = atoi(argv[2]);
    double l = atof(argv[3]);
    double nu = atof(argv[4]);
    double final_time = atof(argv[5]);

    // start timer
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<double> > solution = spectral_heat(p, num_elem, l, nu, final_time);

    // stop timer
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    //std::cout << "Elapsed time: " << elapsed.count() << " s" << std::endl;

    std::vector<double> y = solution[0]; // u_temp
    std::vector<double> x = solution[1]; // x_coord

    
    // write solution to file as x and y
    std::ofstream file;
    file.open("solution.txt");
    for (int i = 0; i < x.size(); i++) {
        file << x[i] << " " << y[i] << std::endl;
    }
    file.close();

    // write elapsed time to file "time.txt"
    std::ofstream file2;
    file2.open("time.txt");
    file2 << elapsed.count() << std::endl;
    file2.close();


    // run python script to plot solution "plot.py"
    system("python3 plot.py");


    return 0;
}

std::vector<std::vector<double> > inverse(std::vector<std::vector<double> > m) {
    int dof = m.size();
    std::vector<std::vector<double> > inv(dof, std::vector<double>(dof));
    for (int i = 0; i < dof; i++) {
        for (int j = 0; j < dof; j++) {
            inv[i][j] = m[i][j];
        }
    }

    for (int i = 0; i < dof; i++) {
        inv[i][i] = 1.0 / inv[i][i];
        for (int j = 0; j < dof; j++) {
            if (i != j) {
                inv[i][j] = -inv[i][j] / inv[i][i];
            }
        }
    }

    return inv;
}

std::vector<double> initial_distribution(std::vector<double> x) {
    std::vector<double> distribution(x.size());
    std::cout << std::fixed << std::setprecision(4);
    for (int i = 0; i < x.size(); i++) {
        distribution[i] = sin((PI) * (1/(x[x.size() - 1] - x[0])) * x[i]);
    }
    return distribution;
}

double lagrange_basis_derivative(std::vector<double> quad_points, int n, double x) {
    double value = 0.0;
    // quad_point - predefined quadrature points
    // n - number of basis function
    // x - point where we want to calculate derivative
    for (int i = 1; i <= quad_points.size(); i++) {
        if (i != n) {
            
            double product = 1.0;
            for (int j = 1; j <= quad_points.size(); j++) {
                if (j != i && j != n) {
                    product = product * (x - quad_points[j-1]) / (quad_points[n-1] - quad_points[j-1]);
                }
            }
            value += 1 / (quad_points[n-1] - quad_points[i-1]) * product;
        }
    }
    return value;
}

double lagrange_basis_polynomial(std::vector<double> quad_points, int count_i, double x) {
    double value_at_xi = 1.0;
    for (int i = 1; i <= quad_points.size(); i++) {
        if (i != count_i) {
            value_at_xi = value_at_xi * (x - quad_points[i-1]) / (quad_points[count_i-1] - quad_points[i-1]);
        }
    }
    return value_at_xi;
}

std::vector<std::vector<double> > lglnodes(int n) {
    std::vector<std::vector<double> > result(2, std::vector<double>(n+1));
    // truncation + 1
    int count_i = n + 1;
   
   // chebyshev-gauss-lobatto nodes
    std::vector<double> x(count_i);

    std::cout << std::fixed << std::setprecision(4);

    for (int i = 0; i < count_i; i++) {
        x[i] = cos(PI * i / n);
    }


    // The Legendre Vandermonde Matrix
    std::vector<std::vector<double> > P(count_i, std::vector<double>(count_i));
    // zeros
    for (int i = 0; i < count_i; i++) {
        for (int j = 0; j < count_i; j++) {
            P[i][j] = 0.0;
        }
    }

    
    // compute p_(n-1) using the recurrence relation
    // compute its first and second derivatives and update
    // x using the Newton-Raphson method
    int x_old = 2;
    std::vector<double> x_old_vector;
    double max_value = 0.0;

    for (int j = 0; j < count_i; j++) {
            max_value = std::max(max_value, std::abs(x[j] - x_old));
    }


    while (max_value > 2.2204e-16) {
        x_old_vector = x;

        for (int j = 0; j < count_i; j++) {
            P[j][0] = 1.0;
        }

        for (int j = 0; j < count_i; j++) {
            P[j][1] = x[j];
        }

        for (int k = 2; k < count_i; k++) {
            for (int j = 0; j < count_i; j++) {
                P[j][k] = ((2.0 * k - 1.0) * x[j] * P[j][k - 1] - (k - 1.0) * P[j][k - 2]) / k;
            }
        }
    
       for (int j = 0; j < count_i; j++) {
            x[j] = x_old_vector[j] - (x[j] * P[j][count_i-1] - P[j][count_i - 2]) / (count_i * P[j][count_i-1]);
        }


        // max value in x
        max_value = 0.0;
        for (int j = 0; j < count_i; j++) {
            max_value = std::max(max_value, std::abs(x[j] - x_old_vector[j]));
        }
    }

    // weights
    std::vector<double> w(count_i);
    for (int i = 0; i < count_i; i++) {
        w[i] = 2.0 / (n * count_i * P[i][count_i-1] * P[i][count_i-1]);
    }

    // sort x in increasing order
    std::sort(x.begin(), x.end());
    result[0] = x;
    result[1] = w;
    return result;
}

// p - order of the polynomial basis function
// num_elements - number of elements in partition
// l - length of the rod
// nu - viscosity
// returns [u_t, x_coord]
std::vector<std::vector<double> > spectral_heat(int p, int num_elem, double l, double nu, double final_time) {
    // partition of the domain
    double delta_x = l / num_elem;
    double delta_t = 0.5 * std::min((delta_x * delta_x) / ((nu + 1)*std::pow(p,4)), final_time/2);
    
    // x_coord = 0 : delta_x : l
    int num_quad_points = p + 1;
    int num_basis_functions = num_quad_points;

    // weights and GL points *[-1, 1]
    // use lglnodes(p) to get the GL points
    std::vector<std::vector<double> > x_w = lglnodes(p);
    std::vector<double> gl_pts = x_w[0];
    std::vector<double> gl_wts = x_w[1];

    // construct the basis lagrange polynomials
    std::vector<std::vector<double> > phi_hat(num_quad_points, std::vector<double>(num_quad_points, 0.0));
    std::vector<std::vector<double> > phi_hat_deriv(num_quad_points, std::vector<double>(num_quad_points, 0.0));
    // for (int i = 0; i < num_quad_points; i++) {
    //     for (int j = 0; j < num_quad_points; j++) {
    //         phi_hat[i][j] = 0.0;
    //         phi_hat_deriv[i][j] = 0.0;
    //     }
    // }
    
    for (int i = 0; i < num_quad_points; i++) {
        for (int j = 0; j < num_quad_points; j++) {
            phi_hat[i][j] = lagrange_basis_polynomial(gl_pts, i+1, gl_pts[j]);
            phi_hat_deriv[i][j] = lagrange_basis_derivative(gl_pts, i+1, gl_pts[j]);
        }
    }



    int dof = num_elem + 1 + (p - 1) * num_elem;
    
    // mass matrix
    std::vector<std::vector<double> > m(dof, std::vector<double>(dof, 0.0));
    // stiffness matrix
    std::vector<std::vector<double> > k(dof, std::vector<double>(dof, 0.0));
    // load vector
    std::vector<std::vector<double> > f(dof, std::vector<double>(1, 0.0));
    // solution vector
    std::vector<std::vector<double> > u_temp(dof, std::vector<double>(1, 0.0));
    
    // for (int i = 0; i < dof; i++) {
    //     for (int j = 0; j < dof; j++) {
    //         m[i][j] = 0.0;
    //         k[i][j] = 0.0;
    //     }
    //     f[i][0] = 0.0;
    //     u_temp[i][0] = 0.0;
    // }

    x_w = lglnodes(p);
    gl_pts = x_w[0];
    gl_wts = x_w[1];

    std::vector<double> unital_gl_pts(num_basis_functions);
    for (int i = 0; i < num_basis_functions; i++) {
        unital_gl_pts[i] = 0.5 * (gl_pts[i] + 1);
    }


    // x_coord
    std::vector<double> x_coord(dof);
    for (int count_temp = 0; count_temp < num_elem; count_temp++) {
        for (int i = (p * (count_temp) + 1)-1, j = 0; i < (p * (count_temp) + 1 + (p - 1)); i++, j++) {
            x_coord[i] = delta_x * ((count_temp) * 1 + unital_gl_pts[j]);
        }
    }
    x_coord[dof-1] = l;


    // compute jacobian matrix on the transformation T
    double j_mat = delta_x / 2;
    double det_j_mat = j_mat;
    double inv_j_mat = 1.0 / j_mat;

    
    // compute the mass matrix
    for (int i = 0; i < num_elem; i++) {
        
        // Me zeros
        std::vector<std::vector<double> > me(num_basis_functions, std::vector<double>(num_basis_functions, 0.0));
        
        // for (int j = 0; j < num_basis_functions; j++) {
        //     for (int k = 0; k < num_basis_functions; k++) {
        //         me[j][k] = 0.0;
        //     }
        // }

        for (int q = 0; q < num_quad_points; q++) {
            double weight = det_j_mat * gl_wts[q];
            for (int j = 0; j < num_basis_functions; j++) {
                for (int k = 0; k < num_basis_functions; k++) {
                    me[j][k] += weight * phi_hat[j][q] * phi_hat[k][q];
                }
            }
        }

       double lbound = i * (num_quad_points-1) + 1;
       double ubound = lbound + num_quad_points - 1;


        for (int j = lbound - 1, i = 0; j < ubound; i++,j++) {
            for (int k = lbound - 1, z = 0; k < ubound; z++,k++) {
                m[j][k] += me[i][z];
            }
        }
        
    }

    
    // create stiffness matrix
    for (int i = 0; i < num_elem; i++) {
            
            // Ke zeros
            std::vector<std::vector<double> > ke(num_basis_functions, std::vector<double>(num_basis_functions, 0.0));
            // for (int j = 0; j < num_basis_functions; j++) {
            //     for (int k = 0; k < num_basis_functions; k++) {
            //         ke[j][k] = 0.0;
            //     }
            // }

        
            for (int q = 0; q < num_quad_points; q++) {
                double weight = det_j_mat * gl_wts[q];
                for (int j = 0; j < num_basis_functions; j++) {
                    for (int z = 0; z < num_basis_functions; z++) {
                        ke[j][z] += phi_hat_deriv[j][q] * inv_j_mat * phi_hat_deriv[z][q] * weight * inv_j_mat;
                    }
                }
            }
            
            
            double lbound = i * (num_quad_points - 1) + 1;
            double ubound = lbound + num_quad_points - 1;

            
            for (int j = lbound - 1, g = 0; j < ubound; g++,j++) {
                
                for (int z = lbound - 1, h = 0; z < ubound; h++,z++) {
                    
                    k[j][z] += ke[g][h];
                }
            }
            
    }

    std::vector<double> u_old(dof);
    std::vector<double> u_new(dof);
    u_old = initial_distribution(x_coord);


    // matrix with 1 on the diagonal
    std::vector<std::vector<double> > diag(dof, std::vector<double>(dof));
    for (int i = 0; i < dof; i++) {
        for (int j = 0; j < dof; j++) {
            if (i == j) {
                diag[i][j] = 1.0;
            } else {
                diag[i][j] = 0.0;
            }
        }
    }

    // inverse matrix
    std::vector<std::vector<double> > inv_m(dof, std::vector<double>(dof));

    //cublas?
    inv_m = inverse(m);

    // multiply nu and delta_t to inv_m
    for (int i = 0; i < dof; i++) {
        for (int j = 0; j < dof; j++) {
            inv_m[i][j] = inv_m[i][j] * nu * delta_t;
        }
    }

    // matrix multiplication of inv_m's columns and k's rows
    std::vector<std::vector<double> > m_inv_k(dof, std::vector<double>(dof));
    for (int i = 0; i < dof; i++) {
        for (int j = 0; j < dof; j++) {
            m_inv_k[i][j] = 0.0;
            for (int z = 0; z < dof; z++) {
                m_inv_k[i][j] += inv_m[i][z] * k[z][j];
            }
        }
    }

    // subtract m_inv_k from diag
    for (int i = 0; i < dof; i++) {
        for (int j = 0; j < dof; j++) {
            diag[i][j] -= m_inv_k[i][j];
        }
    }
    

    for (int count_z = 0; count_z < (final_time / delta_t)-1; count_z++) {

        // matrix vector multiplication of diag and u_old into u_new
        for (int i = 0; i < dof; i++) {
            u_new[i] = 0.0;
            for (int j = 0; j < dof; j++) {
                u_new[i] += diag[i][j] * u_old[j];
            }
        }

        u_new[0] = 0.0;
        u_new[dof-1] = 0.0;
        u_old = u_new;
    }

    // return u_new and x_coord as a vector of vectors
    std::vector<std::vector<double> > result(2);
    result[0] = u_new;
    result[1] = x_coord;

    return result;
}
