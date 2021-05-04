// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h>
using namespace arma;

// Define output structures
struct sub_dog_out{
  arma::mat D;
  arma::mat Gamma;
  arma::mat Omega;
};

struct ADMM_block_out{
  arma::mat om1;
  arma::mat om2;
  arma::mat om3;
  arma::mat gam1;
  arma::mat gam2;
  arma::mat D;
  arma::mat omega;
  arma::mat gamma;
  arma::mat Atilde;
  arma::mat C;
  arma::mat u1;
  arma::mat u2;
  arma::mat u3;
  arma::mat u4;
  arma::mat u5;
  arma::mat omP;
};

struct sub_doc_out{
  arma::mat D;
  arma::mat C;
  arma::mat Omega;
};

struct ADMM_block_out_clusterglasso{
  arma::mat om1;
  arma::mat om2;
  arma::mat c1;
  arma::mat c2;
  arma::mat c3;
  arma::mat D;
  arma::mat omega;
  arma::mat c;
  arma::mat u1;
  arma::mat u2;
  arma::mat u3;
  arma::mat u4;
  arma::mat u5;

};


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////// taglasso functions ///////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

arma::mat refit_omega_ed_sym(const arma::mat& S, const arma::mat& Omega, const arma::mat& U, const double& rho){

  arma::mat rhs = rho * Omega - U - S;
  rhs = (rhs + rhs.t())/2;

  vec eigval;
  mat eigvec;

  arma::eig_sym(eigval, eigvec, rhs);

  arma::mat Omegabar =  arma::diagmat((eigval + sqrt(square(eigval) + 4 * rho) ) / (2*rho));
  arma::mat Omeganew = eigvec * Omegabar *eigvec.t();

  return(Omeganew);
}

arma::mat refit_omega_soft(const arma::mat& Omega, const arma::mat& U, const double& rho, const arma::mat& omP){
  // Input
  // Omega : matrix of dimension p times p
  // U : dual variable, matrix of dimension p times p
  // rho : parameter ADMM
  // omP : matrix of dimension p times p with 0 or 1 as entries: 0 (zero-elements) and 1 (non-zero elements)

  // Function : Obtain estimate of Omega^(3)

  // Output
  // Omeganew : matrix of dimension p times p

  int p = Omega.n_cols;
  arma::mat Omeganew = zeros(p, p);
  arma::mat soft_input = Omega - U/rho;

  arma::vec diago = zeros(p);

  for(int ir=0; ir < p; ++ir ){
    for(int ic=0; ic < p; ++ ic){
      if(omP(ir, ic)==0){
        Omeganew(ir, ic) = 0;
      }else{
        Omeganew(ir, ic) = soft_input(ir, ic);
      }
    }
  }

  return(Omeganew);
}

double softelem(const double& a, const double& lambda){
  // Input
  // a : scalar
  // lambda : scalar, tuning parameter

  // Function : Elementwise soft-thresholding (needed in function solve_omega_soft)

  // Output
  // out : scalar after elementwise soft-thresholding

  double out = ((a > 0) - (a < 0)) * std::max(0.0, std::abs(a) - lambda);
  return(out);
}

arma::mat solve_omega_soft(const arma::mat& Omega, const arma::mat& U, const double& rho, const double& lambda,
                           const bool& pendiag){
  // Input
  // Omega : matrix of dimension p times p
  // U : dual variable, matrix of dimension p times p
  // rho : parameter ADMM
  // lambda : regularization parameter for elementwise soft thresholding
  // pendiag : logical, penalize diagonal or not

  // Function : Obtain estimate of Omega^(1) (4.1.1)

  // Output
  // Omeganew : matrix of dimension p times p

  int p = Omega.n_cols;
  arma::mat Omeganew = zeros(p, p);
  arma::mat soft_input = Omega - U/rho;

  arma::vec diago = zeros(p);

  for(int ir=0; ir < p; ++ir ){
    for(int ic=0; ic < p; ++ ic){
      Omeganew(ir, ic) = softelem(soft_input(ir, ic), lambda/rho);
    }
  }

  if(!pendiag){
    Omeganew.diag() = arma::diagvec(Omega - U/rho);
  }

  return(Omeganew);
}

arma::vec softgroup(const arma::vec& u, const double& lambda){
  // Input
  // u : vector (of dimension p)
  // lambda : scalar, tuning parameter

  // Function : Groupwise soft-thresholding (needed in function solve_gamma_soft)

  // Output
  // sg : vector after groupwise soft-thresholding

  arma::vec sg = std::max(0.0, 1 - lambda/arma::norm(u,2))*u;
  return(sg);
}

arma::mat solve_gamma_soft(const arma::mat& Gamma, const arma::mat& U, const double& rho, const double& lambda){
  // Input
  // Gamma : matrix of dimension |T| times p
  // U : dual variable, matrix of dimension |T| times p
  // rho : parameter ADMM
  // lambda : regularization parameter for groupwise soft-thresholding

  // Function : Obtain estimate of Gamma^(1)

  // Output
  // Gammanew : matrix of dimension |T| times p

  int t = Gamma.n_rows;
  int p = Gamma.n_cols;
  arma::mat Gammanew = zeros(t, p);
  arma::mat soft_input = Gamma - U/rho;
  arma::rowvec outt = soft_input.row(0);

  if(lambda==0){
    Gammanew = Gamma - U/rho;  // no shrinkage
    Gammanew.row(t-1).fill(mean(Gammanew.row(t-1)));
  }else{

    for(int ir=0; ir < t; ++ir){
      outt = soft_input.row(ir);
      if(ir == (t-1)){
        Gammanew.row(ir).fill(mean(outt));
      }else{
        Gammanew.row(ir) = softgroup(outt.t(), lambda/rho).t();
      }
    }
  }

  return(Gammanew);
}

sub_dog_out solve_DOG(const arma::mat& A, const arma::mat& Omega, const arma::mat& Uom, const arma::mat& Gamma,
                      const arma::mat& Ugam, const double& rho, const arma::mat& Atilde, const arma::mat& A_for_gamma,
                      const arma::mat& A_for_B, const arma::mat& C, const arma::mat& C_for_D){
  // Input
  // A : matrix of dimension p times |T|
  // Omega : matrix of dimension p times p
  // Uom: dual variable, matrix of dimension p times p
  // Gamma : matrix of dimension |T| times p
  // Ugam : dual variable, matrix of dimension |T| times p
  // rho : parameter ADMM
  // Atilde : rbind(A, I_|T|x|T|)
  // A_for_gamma : matrix of dimension |T|x|T|
  // A_for_B : matrix of dimension dimension (p+|T|)x(p+|T|)
  // C : matrix of dimension (p+ |T|) times p (requires Atilde)
  // C_for_D : matrix of dimension p times p

  // Function : Obtain estimates of Omega^(2), Gamma^(2) and D

  // Output
  // D : matrix of dimension p times p
  // Gamma: matrix of dimension |T| times p
  // Omega : matrix of dimension p times p

  int p = Omega.n_cols;
  int t = A.n_cols;

  // solve for D
  arma::mat Mtilde = arma::join_cols( Omega - Uom / rho, Gamma - Ugam/rho );
  arma::mat B = A_for_B * Mtilde;
  arma::vec BCd = arma::diagvec(B.t() * C);
  arma::mat BCdm = max(BCd, zeros(p));
  arma::mat BCdm2 = arma::diagmat(BCdm);
  arma::mat Dnew  = C_for_D * BCdm2;

  // solve for Gamma^(2)
  arma::mat Dtilde = arma::join_cols( Dnew, zeros(t, p) );
  arma::mat Gammanew = A_for_gamma * (Mtilde - Dtilde);

  // solve for Omega^(2)
  arma::mat Omeganew = A * Gammanew + Dnew;

  sub_dog_out dogout;
  dogout.D = Dnew;
  dogout.Gamma = Gammanew;
  dogout.Omega = Omeganew;

  return(dogout);
}

ADMM_block_out ADMM_taglasso_block(const arma::mat& S, const arma::mat& A, const arma::mat& Atilde, const arma::mat& A_for_gamma,
                                   const arma::mat& A_for_B, const arma::mat& C, const arma::mat& C_for_D, const double& lambda1,
                                   const double& lambda2, const double& rho, const bool& pendiag, const double& maxite,
                                   const arma::mat& init_om, const arma::mat& init_u1, const arma::mat& init_u3,
                                   const arma::mat& init_u4, const arma::mat& init_gam, const arma::mat& init_u2,
                                   const arma::mat& init_u5){
  // Input
  // S : sample covariance matrix of dimension p times p
  // A : matrix of dimension p times |T|
  // Atilde : rbind(A, I_|T|x|T|)
  // A_for_gamma : matrix of dimension |T|x|T|
  // A_for_B : matrix of dimension dimension (p+|T|)x(p+|T|)
  // C : matrix of dimension (p+ |T|) times p (requires Atilde)
  // C_for_D : matrix of dimension p times p
  // lambda1 : scalar, regularization parameter lambda1||Gamma^(1)_r||_{2,1}
  // lambda2: scalar, regularization parameter lambda2||Omega||_1
  // rho : scalar, parameter ADMM
  // pendiag : logical, penalize diagonal or not when solving for Omega^(1)
  // maxite : scalar, maximum number of iterations
  // init_om : matrix of dimension p times p, initialization of Omega
  // init_u1 : matrix of dimension p times p, initialization of dual variable U1 of Omega^(1)
  // init_u3 : matrix of dimension p times p, initialization of dual variable U3 of Omega^(3)
  // init_u4 : matrix of dimension |T| times p, initialization of dual variable U4 of Gamma^(1)
  // init_gam : matrix of dimension |T| times p, initialization of Gamma
  // init_u2 : matrix of dimension p times p, initialization of dual variable U2 of Omega^(2)
  // init_u5 : matrix of dimension |T| times p, initialization of dual variable U5 of Gamma^(2)

  // Function : ADMM update

  int p = S.n_cols;
  int nnodes = A.n_cols;

  arma::mat omegaold = init_om;
  arma::mat gammaold = init_gam;

  arma::mat u1 = init_u1;
  arma::mat u2 = init_u2;
  arma::mat u3 = init_u3;
  arma::mat u4 = init_u4;
  arma::mat u5 = init_u5;

  arma::mat om1 = zeros(p, p); // eigenvalue decomposition
  arma::mat om2 = zeros(p, p); // AG+D
  arma::mat om3 = zeros(p, p); // soft thresholding
  arma::mat gam1 = zeros(nnodes, p); // groupwise soft thresholding -->*IW: THIS BECOMES THE NEW SUBPROBLEM*
  arma::mat gam2 = zeros(nnodes, p); // AG+D
  arma::mat d = zeros(p, p); // AG+D

  sub_dog_out dogout_fit;

  for(int iin=0; iin < maxite; ++iin){

    // Solve for Omega^(1) : Eigenvalue decomposition
    om1 = refit_omega_ed_sym(S, omegaold, u1, rho); // output is a matrix of dimension p times p

    // Solve for Omega^(3): Soft-thresholding
    om3 = solve_omega_soft(omegaold, u3, rho, lambda2, pendiag); // output is a matrix of dimension p times p

    // Solve for Gamma^(1) : Groupwise soft-thresholding -->*IW: THIS BECOMES THE NEW SUBPROBLEM*
    gam1 = solve_gamma_soft(gammaold, u4, rho, lambda1); // output is a matrix of dimension |T| times p

    // Solve for D, Omega^(2) and Gamma^(2)
    dogout_fit = solve_DOG(A, omegaold, u2, gammaold, u5, rho, Atilde, A_for_gamma, A_for_B, C, C_for_D); // output is a List
    om2 = dogout_fit.Omega;
    gam2 = dogout_fit.Gamma;
    d = dogout_fit.D;

    // Updating Omega, Gamma
    omegaold = (om1 + om2 + om3) / 3;
    gammaold = (gam1 + gam2) / 2;

    // Update Dual variables
    u1 = u1 + rho * ( om1 - omegaold);
    u2 = u2 + rho * ( om2 - omegaold);
    u3 = u3 + rho * ( om3 - omegaold);
    u4 = u4 + rho * ( gam1 - gammaold);
    u5 = u5 + rho * ( gam2 - gammaold);
  }

  ADMM_block_out ADMMblockout;
  ADMMblockout.om1 = om1;
  ADMMblockout.om2 = om2;
  ADMMblockout.om3 = om3;
  ADMMblockout.gam1 = gam1;
  ADMMblockout.gam2 = gam2;
  ADMMblockout.D = d;
  ADMMblockout.omega = omegaold;
  ADMMblockout.gamma = gammaold;
  ADMMblockout.Atilde = Atilde;
  ADMMblockout.C = C;
  ADMMblockout.u1 = u1;
  ADMMblockout.u2 = u2;
  ADMMblockout.u3 = u3;
  ADMMblockout.u4 = u4;
  ADMMblockout.u5 = u5;

  return(ADMMblockout);
}


// [[Rcpp::export]]
Rcpp::List LA_ADMM_taglasso_export(const int& it_out, const int& it_in, const arma::mat& S, const arma::mat& A,
                                   const arma::mat& Atilde, const arma::mat& A_for_gamma, const arma::mat& A_for_B,
                                   const arma::mat& C, const arma::mat& C_for_D, const double& lambda1,
                                   const double& lambda2, const double& rho, const bool& pendiag, const arma::mat& init_om,
                                   const arma::mat& init_u1, const arma::mat& init_u2, const arma::mat& init_u3,
                                   const arma::mat& init_gam, const arma::mat& init_u4, const arma::mat& init_u5){
  // Input
  // it_out : scalar, T_stages of LA-ADMM algorithm
  // it_in : scalar, maximum number of iterations of (inner) ADMM algorithm
  // S : sample covariance matrix of dimension p times p
  // A : matrix of dimension p times |T|
  // Atilde : rbind(A, I_|T|x|T|)
  // A_for_gamma : matrix of dimension |T|x|T|
  // A_for_B : matrix of dimension dimension (p+|T|)x(p+|T|)
  // C : matrix of dimension (p+ |T|) times p (requires Atilde)
  // C_for_D : matrix of dimension p times p
  // lambda1 : scalar, regularization parameter lambda1||Gamma^(1)_r||_{2,1}
  // lambda2: scalar, regularization parameter lambda2||Omega||_1
  // rho : scalar, parameter ADMM
  // pendiag : logical, penalize diagonal or not when solving for Omega^(1)
  // init_om : matrix of dimension p times p, initialization of Omega
  // init_u1 : matrix of dimension p times p, initialization of dual variable U1 of Omega^(1)
  // init_u3 : matrix of dimension p times p, initialization of dual variable U3 of Omega^(3)
  // init_u4 : matrix of dimension |T| times p, initialization of dual variable U4 of Gamma^(1)
  // init_gam : matrix of dimension |T| times p, initialization of Gamma
  // init_u2 : matrix of dimension p times p, initialization of dual variable U2 of Omega^(2)
  // init_u5 : matrix of dimension |T| times p, initialization of dual variable U5 of Gamma^(2)

  // Function : LA-ADMM updates

  // Preliminaries
  arma::mat in_om = init_om;
  arma::mat in_gam = init_gam;

  double rhoold = rho;
  double rhonew = rho;

  ADMM_block_out ADMMout;

  for(int iout=0; iout < it_out; ++iout){

    ADMMout = ADMM_taglasso_block(S, A, Atilde, A_for_gamma, A_for_B, C, C_for_D, lambda1, lambda2, rhonew, pendiag, it_in,
                                    in_om, init_u1, init_u3, init_u4, in_gam, init_u2, init_u5);
    in_om  = ADMMout.omega;
    in_gam = ADMMout.gamma;
    rhonew = 2*rhoold;
    rhoold = rhonew;
  }

  // Remark: We don't need to return everything for final version, but can be useful now
  Rcpp::List results=Rcpp::List::create(
    Rcpp::Named("omega") = ADMMout.omega,
    Rcpp::Named("gamma") = ADMMout.gamma,
    Rcpp::Named("om1") = ADMMout.om1,
    Rcpp::Named("om2") = ADMMout.om2,
    Rcpp::Named("om3") = ADMMout.om3,
    Rcpp::Named("gam1") = ADMMout.gam1,
    Rcpp::Named("gam2") = ADMMout.gam2,
    Rcpp::Named("D") = ADMMout.D,
    Rcpp::Named("u1") = ADMMout.u1,
    Rcpp::Named("u2") = ADMMout.u2,
    Rcpp::Named("u3") = ADMMout.u3,
    Rcpp::Named("u4") = ADMMout.u4,
    Rcpp::Named("u5") = ADMMout.u5,
    Rcpp::Named("rho") = rhonew);

  return(results);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////// refit functions for taglasso /////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

arma::mat refit_gamma_soft(const arma::mat& Gamma, const arma::mat& U, const double& rho){
    // Input
    // Gamma : matrix of dimension |Z| times p
    // U : dual variable, matrix of dimension |Z| times p
    // rho : parameter ADMM

    // Function : Obtain estimate of Gamma_Z^(1)

    // Output
    // Gammanew : matrix of dimension |Z| times p

    int z = Gamma.n_rows;
    int p = Gamma.n_cols;
    arma::mat Gammanew = zeros(z, p);

    Gammanew = Gamma - U/rho;
    Gammanew.row(z-1).fill(mean(Gammanew.row(z-1)));

    return(Gammanew);
  }

sub_dog_out refit_DOG(const arma::mat& A, const arma::mat& Omega, const arma::mat& Uom, const arma::mat& Gamma,
                      const arma::mat& Ugam, const double& rho, const arma::mat& Atilde, const arma::mat& A_for_gamma,
                      const arma::mat& A_for_B, const arma::mat& C, const arma::mat& C_for_D){
  // Input
  // A : matrix of dimension p times |Z|
  // Omega : matrix of dimension p times p
  // Uom: dual variable, matrix of dimension p times p
  // Gamma : matrix of dimension |Z| times p
  // Ugam : dual variable, matrix of dimension |Z| times p
  // rho : parameter ADMM
  // Atilde : rbind(A, I_|Z|x|Z|)
  // A_for_gamma : matrix of dimension |Z|x|Z|
  // A_for_B : matrix of dimension dimension (p+|Z|)x(p+|Z|)
  // C : matrix of dimension (p+ |Z|) times p (requires Atilde)
  // C_for_D : matrix of dimension p times p

  // Function : Obtain estimates of Omega^(2), Gamma_Z^(2) and D

  // Output
  // D : matrix of dimension p times p
  // Gamma: matrix of dimension |Z| times p
  // Omega : matrix of dimension p times p

  int p = Omega.n_cols;
  int z = A.n_cols;

  // solve for D
  arma::mat Mtilde = arma::join_cols( Omega - Uom / rho, Gamma - Ugam/rho );
  arma::mat B = A_for_B * Mtilde;
  arma::vec BCd = arma::diagvec(B.t() * C);
  arma::mat BCdm = max(BCd, zeros(p));
  arma::mat BCdm2 = arma::diagmat(BCdm);
  arma::mat Dnew  = C_for_D * BCdm2;

  // solve for Gamma^(2)
  arma::mat Dtilde = arma::join_cols( Dnew, zeros(z, p) );
  arma::mat Gammanew = A_for_gamma * (Mtilde - Dtilde);

  // solve for Omega^(2)
  arma::mat Omeganew = A * Gammanew + Dnew;

  sub_dog_out dogout;
  dogout.D = Dnew;
  dogout.Gamma = Gammanew;
  dogout.Omega = Omeganew;
  return dogout;
}

ADMM_block_out refit_ADMM_block_new(const arma::mat& S, const arma::mat& A, const arma::mat& Atilde, const arma::mat& A_for_gamma,
                                    const arma::mat& A_for_B, const arma::mat& C, const arma::mat& C_for_D, const double& rho,
                                    const arma::mat& omP, double maxite, const arma::mat& init_om, const arma::mat& init_u1,
                                    const arma::mat& init_u2, const arma::mat& init_u3, const arma::mat& init_gam,
                                    const arma::mat& init_u4, const arma::mat& init_u5){
  // Input
  // S : sample covariance matrix of dimension p times p
  // A : matrix of dimension p times |Z|
  // Atilde : rbind(A, I_|Z|x|Z|)
  // A_for_gamma : matrix of dimension |Z|x|Z|
  // A_for_B : matrix of dimension dimension (p+|Z|)x(p+|Z|)
  // C : matrix of dimension (p+ |Z|) times p (requires Atilde)
  // C_for_D : matrix of dimension p times p
  // rho : scalar, parameter ADMM
  // omP : matrix of dimension p times p with 0 or 1 as entries: 0 (zero-elements) and 1 (non-zero elements)
  // maxite : scalar, maximum number of iterations
  // init_om : matrix of dimension p times p, initialization of Omega
  // init_u1 : matrix of dimension p times p, initialization of dual variable U1 of Omega^(1)
  // init_u2 : matrix of dimension p times p, initialization of dual variable U2 of Omega^(2)
  // init_u3 : matrix of dimension p times p, initialization of dual variable U3 of Omega^(3)
  // init_gam : matrix of dimension |Z| times p, initialization of Gamma
  // init_u4 : matrix of dimension |Z| times p, initialization of dual variable U4 of Gamma^(1)
  // init_u5 : matrix of dimension |Z| times p, initialization of dual variable U5 of Gamma^(2)

  // Function : ADMM update

  int p = S.n_cols;
  int nnodes = A.n_cols;

  arma::mat omegaold = init_om;
  arma::mat gammaold = init_gam;

  arma::mat u1 = init_u1;
  arma::mat u2 = init_u2;
  arma::mat u3 = init_u3;
  arma::mat u4 = init_u4;
  arma::mat u5 = init_u5;

  arma::mat om1 = zeros(p, p); // eigenvalue decomposition
  arma::mat om2 = zeros(p, p); // AG+D
  arma::mat om3 = zeros(p, p); // soft- thresholding
  arma::mat gam1 = zeros(nnodes, p); // groupwise soft-thresholding
  arma::mat gam2 = zeros(nnodes, p); // AG+D
  arma::mat d = zeros(p, p); // AG+D

  sub_dog_out dogout_fit;

  for(int iin=0; iin < maxite; ++iin){
    // Solve for Omega^(1) : Eigenvalue decomposition
    om1 = refit_omega_ed_sym(S, omegaold, u1, rho); // output is a matrix of dimension p times p

    // Solve for Omega^(3) : Soft-thresholding
    om3 = refit_omega_soft(omegaold, u3, rho, omP); // output is a matrix of dimension p times p

    // Solve for Gamma_Z^(1) : Groupwise soft-thresholding
    gam1 = refit_gamma_soft(gammaold, u4, rho); // output is a matrix of dimension |Z| times p

    // Solve for D, Omega^(2) and Gamma_Z^(2)
    dogout_fit = refit_DOG(A, omegaold, u2, gammaold, u5, rho, Atilde, A_for_gamma, A_for_B, C, C_for_D); // output is a List
    om2 = dogout_fit.Omega;
    gam2 = dogout_fit.Gamma;
    d = dogout_fit.D;

    // Updating Omega and Gamma_Z
    omegaold = (om1 + om2 + om3) / 3;
    gammaold = (gam1 + gam2) / 2;

    // Update Dual variables
    u1 = u1 + rho * ( om1 - omegaold);
    u2 = u2 + rho * ( om2 - omegaold);
    u3 = u3 + rho * ( om3 - omegaold);
    u4 = u4 + rho * ( gam1 - gammaold);
    u5 = u5 + rho * ( gam2 - gammaold);

  }

  ADMM_block_out ADMMblockout;
  ADMMblockout.om1 = om1;
  ADMMblockout.om2 = om2;
  ADMMblockout.om3 = om3;
  ADMMblockout.gam1 = gam1;
  ADMMblockout.gam2 = gam2;
  ADMMblockout.D = d;
  ADMMblockout.omega = omegaold;
  ADMMblockout.gamma = gammaold;
  ADMMblockout.Atilde = Atilde;
  ADMMblockout.C = C;
  ADMMblockout.u1 = u1;
  ADMMblockout.u2 = u2;
  ADMMblockout.u3 = u3;
  ADMMblockout.u4 = u4;
  ADMMblockout.u5 = u5;
  ADMMblockout.omP = omP;
  return(ADMMblockout);
}

// [[Rcpp::export]]
Rcpp::List refit_LA_ADMM_export(const int& it_out, const int& it_in , const arma::mat& S, const arma::mat& A,
                                const arma::mat& Atilde, const arma::mat& A_for_gamma, const arma::mat& A_for_B,
                                const arma::mat& C, const arma::mat& C_for_D, const double& rho, const arma::mat& omP,
                                const arma::mat& init_om, const arma::mat& init_u1, const arma::mat& init_u2,
                                const arma::mat& init_u3, const arma::mat& init_gam, const arma::mat& init_u4,
                                const arma::mat& init_u5){
  // Input
  // it_out : scalar, T_stages of LA-ADMM algorithm
  // it_in : scalar, maximum number of iterations of ADMM algorithm
  // S : sample covariance matrix of dimension p times p
  // A : matrix of dimension p times |Z|
  // Atilde : rbind(A, I_|Z|x|Z|)
  // A_for_gamma : matrix of dimension |Z|x|Z|
  // A_for_B : matrix of dimension dimension (p+|Z|)x(p+|Z|)
  // C : matrix of dimension (p+ |Z|) times p (requires Atilde)
  // C_for_D : matrix of dimension p times p
  // rho : scalar, parameter ADMM
  // omP : matrix of dimension p times p with 0 or 1 as entries: 0 (zero-elements) and 1 (non-zero elements)
  // init_om : matrix of dimension p times p, initialization of Omega
  // init_u1 : matrix of dimension p times p, initialization of dual variable U1 of Omega^(1)
  // init_u2 : matrix of dimension p times p, initialization of dual variable U2 of Omega^(2)
  // init_u3 : matrix of dimension p times p, initialization of dual variable U3 of Omega^(3)
  // init_gam : matrix of dimension |Z| times p, initialization of Gamma
  // init_u4 : matrix of dimension |Z| times p, initialization of dual variable U4 of Gamma^(1)
  // init_u5 : matrix of dimension |Z| times p, initialization of dual variable U5 of Gamma^(2)

  // Function : LA-ADMM updates

  // Preliminaries
  arma::mat in_om = init_om;
  arma::mat in_gam = init_gam;

  double rhoold = rho;
  double rhonew = rho;
  ADMM_block_out ADMMblockout_fit;

  for(int iout=0; iout < it_out; ++iout){
    ADMMblockout_fit = refit_ADMM_block_new(S, A, Atilde, A_for_gamma, A_for_B, C, C_for_D, rhonew , omP, it_in, in_om, init_u1, init_u2, init_u3,
                                            in_gam, init_u4, init_u5);

    in_om  = ADMMblockout_fit.omega;
    in_gam = ADMMblockout_fit.gamma;

    rhonew = 2*rhoold;
    rhoold = rhonew;
  }

  Rcpp::List results=Rcpp::List::create(
    Rcpp::Named("omega") = ADMMblockout_fit.omega,
    Rcpp::Named("gamma") = ADMMblockout_fit.gamma,
    Rcpp::Named("om1") = ADMMblockout_fit.om1,
    Rcpp::Named("om2") = ADMMblockout_fit.om2,
    Rcpp::Named("om3") = ADMMblockout_fit.om3,
    Rcpp::Named("gam1") = ADMMblockout_fit.gam1,
    Rcpp::Named("gam2") = ADMMblockout_fit.gam2,
    Rcpp::Named("D") = ADMMblockout_fit.D,
    Rcpp::Named("u1") = ADMMblockout_fit.u1,
    Rcpp::Named("u2") = ADMMblockout_fit.u2,
    Rcpp::Named("u3") = ADMMblockout_fit.u3,
    Rcpp::Named("u4") = ADMMblockout_fit.u4,
    Rcpp::Named("u5") = ADMMblockout_fit.u5,
    Rcpp::Named("omP") = omP,
    Rcpp::Named("rho") = rhonew);

  return(results);
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////// NEW FUNCTIONS FOR CLUSTER GLASSO //////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


sub_doc_out solve_DOC(const arma::mat& A, const arma::mat& Omega, const arma::mat& Uom, const arma::mat& C,
                      const arma::mat& Uc, const double& rho,
                      const arma::mat& Itilde, const arma::mat& A_for_C3, const arma::mat& A_for_T1, const arma::mat& T2, const arma::mat& T2_for_D){
  // Input
  // A : matrix of dimension p times p. In our case this is the identity matrix since we have (in the format of the old code) Omega = AC+D with A=I_pxp
  // Omega : matrix of dimension p times p
  // Uom: dual variable, matrix of dimension p times p
  // C : matrix of dimension p times p
  // Uc : dual variable, matrix of dimension p times p
  // rho : parameter ADMM

  // Itilde : rbind(A, I_pxp)
  // A_for_C3 : matrix of dimension px(2p)
  // A_for_T1 : matrix of dimension dimension (p+p)x(p+p)
  // T2 : matrix of dimension (p+p) times p (requires Itilde)
  // T2_for_D : matrix of dimension p times p
  // Note that all these matrices can be computed at the start (now also in a more efficient way since A is the identity matrix; but I haven't changed this yet); we need them in the subproblem when we solve for Omega^(2), C^(3) and D

  // Function : Obtain estimates of Omega^(2), C^(3) and D

  // Output
  // D : matrix of dimension p times p
  // C: matrix of dimension p times p
  // Omega : matrix of dimension p times p

  int p = Omega.n_cols;

  // solve for D
  arma::mat Rtilde = arma::join_cols( Omega - Uom / rho, C - Uc/rho );
  arma::mat T1 = A_for_T1 * Rtilde;
  arma::vec T1T2d = arma::diagvec(T1.t() * T2);
  arma::mat T1T2dm = max(T1T2d, zeros(p));
  arma::mat T1T2m = arma::diagmat(T1T2dm);
  arma::mat Dnew  = T2_for_D * T1T2m;

  // solve for Gamma^(2)
  arma::mat Dtilde = arma::join_cols( Dnew, zeros(p, p) );
  arma::mat Cnew = A_for_C3 * (Rtilde - Dtilde);

  // solve for Omega^(2)
  arma::mat Omeganew = Cnew + Dnew;

  sub_doc_out docout;
  docout.D = Dnew;
  docout.C = Cnew;
  docout.Omega = Omeganew;

  return(docout);
}


double cluster_loss(const arma::mat& c2, const arma::mat& X, const arma::mat& W, const double& rho, const double& lambda2)
{
  // Input
  // c2 : matrix of dimensions p times p which we use to minimize the loss
  // X : matrix of dimensions p times p which is the "anchor point" in the minimization. In our case this X = cold - u5 / rho
  // W : matrix of dimensions p times p which contains the weights used to reflect importance of clustering i and j
  // rho : scalar, parameter ADMM
  // lambda2 : scalar, regularization parameter clustering

  // Function : Compute the convex clustering loss for a given c2

  // Output
  // result : scalar, loss

  // Preliminary
  int p = c2.n_rows;

  // Compute first term of the loss function and initialize the value for the penalty as zero
  double term1 = rho / 2 * pow(arma::norm(c2 - X, "fro"), 2.0);
  double penalty = 0.0;

  // Compute the penalty
  for (int i = 0; i < p; i++) {
    for (int j = 0; j < i; j++) {
      penalty += W(i, j) * arma::norm(c2.row(i) - c2.row(j));
    }
  }

  // Add both terms
  double result = term1 + lambda2 * penalty;

  return result;
}


void compute_update(arma::mat& c2, const arma::mat& X, const arma::mat& W, const double& rho, const double& lambda2)
{
  // Input
  // c2 : matrix of dimensions p times p which we use to minimize the loss
  // X : matrix of dimensions p times p which is the "anchor point" in the minimization. In our case this X = cold - u5 / rho
  // W : matrix of dimensions p times p which contains the weights used to reflect importance of clustering i and j
  // rho : scalar, parameter ADMM
  // lambda2 : scalar, regularization parameter clustering

  // Function : Computes the update for c2

  // Output
  // void, c2 is passed by reference and adjusted inside the function

  // Preliminaries
  int p = c2.n_rows;
  arma::mat V(p, p, arma::fill::zeros);

  // Fill the matrix V
  for (int i = 0; i < p; i++) {
    for (int j = 0; j < i; j++) {
      double temp = 1 / std::max(arma::norm(c2.row(i) - c2.row(j)), 1e-5);

      V(i, i) += temp;
      V(j, j) += temp;
      V(i, j) -= temp;
      V(j, i) -= temp;
    }
  }

  // Compute the update as c2 = (rho * I_pxp + lambda2 * V)^-1 * rho * X
  V = rho * arma::eye(p, p) + lambda2 * V;
  c2 = arma::solve(V, rho * X);
}


arma::mat FUNCTION_FROM_DANIEL(const arma::mat& cold, const arma::mat& u5, const double& rho, const double& lambda2)
{
  // Input
  // X : matrix of dimensions p times p which is the "anchor point" in the minimization. In our case this X = cold - u5 / rho
  // W : matrix of dimensions p times p which contains the weights used to reflect importance of clustering i and j
  // rho : scalar, parameter ADMM
  // lambda2 : scalar, regularization parameter clustering

  // Function : Computes the update for c2

  // Output
  // c2 : matrix of dimensions p times p which we use to minimize the loss

  // Preliminaries
  int p = cold.n_rows;
  int t = 0;
  arma::mat X = cold - 1 / rho * u5;
  arma::mat c2(X);
  arma::mat W(p, p, arma::fill::ones);

  // Initialize loss values
  double loss1 = cluster_loss(c2, X, W, rho, lambda2);
  double loss0 = 2 * loss1;

  // Set some constants
  const int max_iter = 500;
  const double eps_conv = 1e-7;

  // While the relative decrease in the loss function is above some value and the maximum number of iterations is not reached, update c2
  while (fabs((loss0 - loss1) / loss0) > eps_conv && t < max_iter) {
    compute_update(c2, X, W, rho, lambda2);

    loss0 = loss1;
    loss1 = cluster_loss(c2, X, W, rho, lambda2);

    t++;
  }

  return c2;
}


ADMM_block_out_clusterglasso ADMM_clusterglasso_block(const arma::mat& S,
                                   const arma::mat& A, const arma::mat& Itilde, const arma::mat& A_for_C3, const arma::mat& A_for_T1, const arma::mat& T2, const arma::mat& T2_for_D,
                                   const double& lambda1, const double& lambda2, const double& rho, const bool& pendiag, const double& maxite,
                                   const arma::mat& init_om, const arma::mat& init_u1, const arma::mat& init_u2,
                                   const arma::mat& init_c, const arma::mat& init_u3, const arma::mat& init_u4, const arma::mat& init_u5){
  // Input
  // S : sample covariance matrix of dimension p times p
  // A : matrix of dimension p times p. In our case this is the identity matrix since we have (in the format of the old code) Omega = AC+D with A=I_pxp
  // Itilde : rbind(A, I_pxp)
  // A_for_C3 : matrix of dimension px(2p)
  // A_for_T1 : matrix of dimension dimension (p+p)x(p+p)
  // T2 : matrix of dimension (p+p) times p (requires Itilde)
  // T2_for_D : matrix of dimension p times p
  // Note that all these matrices can be computed at the start (now also in a more efficient way since A is the identity matrix; but I haven't changed this yet); we need them in the subproblem when we solve for Omega^(2), C^(3) and D
  // lambda1 : scalar, regularization parameter sparsity --> *IW: NOTE I CHANGED THIS COMPARED TO THE code.cpp document since on Dropbox in .pdf meetings file we use lambda1 for l1 norm and lambda2 for clustering penalty
  // lambda2: scalar, regularization parameter clustering --> *IW: NOTE I CHANGED THIS COMPARED TO THE code.cpp document since on Dropbox in .pdf meetings file we use lambda1 for l1 norm and lambda2 for clustering penalty
  // rho : scalar, parameter ADMM
  // pendiag : logical, penalize diagonal or not when solving for Omega^(1)
  // maxite : scalar, maximum number of iterations
  // init_om : matrix of dimension p times p, initialization of Omega
  // init_u1 : matrix of dimension p times p, initialization of dual variable U1 of Omega^(1)
  // init_u2 : matrix of dimension p times p, initialization of dual variable U2 of Omega^(2)
  // init_c : matrix of dimension p times p, initialization of C
  // init_u3 : matrix of dimension p times p, initialization of dual variable U3 of C^(3)
  // init_u4 : matrix of dimension p times p, initialization of dual variable U4 of C^(1)
  // init_u5 : matrix of dimension p times p, initialization of dual variable U5 of C^(2)

  // Function : ADMM update

  int p = S.n_cols;

  arma::mat omegaold = init_om;
  arma::mat cold = init_c;

  arma::mat u1 = init_u1;
  arma::mat u2 = init_u2;
  arma::mat u3 = init_u3;
  arma::mat u4 = init_u4;
  arma::mat u5 = init_u5;

  arma::mat om1 = zeros(p, p); // eigenvalue decomposition
  arma::mat om2 = zeros(p, p); // Omega = AC + D (with A the identity matrix)

  arma::mat c1 = zeros(p, p); // elementwise soft thresholding
  arma::mat c2 = zeros(p, p); // groupwise soft thresholding -->*IW: THIS BECOMES THE NEW SUBPROBLEM BASED ON DANIEL's THESIS*
  arma::mat c3 = zeros(p, p); // Omega = AC + D (with A the identity matrix)

  arma::mat d = zeros(p, p); // Omega = AC + D (with A the identity matrix)

  sub_doc_out docout_fit;

  for(int iin=0; iin < maxite; ++iin){

    // Solve for Omega^(1) : Eigenvalue decomposition
    om1 = refit_omega_ed_sym(S, omegaold, u1, rho); // output is an arma::mat matrix of dimension p times p

    // Solve for C^(1): Soft-thresholding
    c1 = solve_omega_soft(cold, u4, rho, lambda1, pendiag); // output is an arma::mat matrix of dimension p times p

    // Solve for C^(2) : Clustering
    // -->*IW: THIS BECOMES THE NEW SUBPROBLEM*
    // --> *IW: INPUT: It should take as inputs cold, u5,  rho (these all appear in the squared frob. norm of the objective function with cold = Chat in my document and u5=uhat5); and on the regularization parameter lambda2 for clustering*
    // --> *IW: More precisely: the objective function is Chat2 = argmin = (rho/2) ||C2 - (cold - u5/rho) ||^2_F + lambda2*Penalty(C2), with Penalty(C2) the penalty from Daniel's thesis
    // --> *IW: OUTPUT: It should have a pxp arma::mat as output*
    // c2 = FUNCTION_FROM_DANIEL(cold, u5, rho, lambda2); // output is a arma::mat matrix of dimension p times p

    // Solve for D, Omega^(2) and C^(3)
    docout_fit = solve_DOC(A, omegaold, u2, cold, u3, rho, Itilde, A_for_C3, A_for_T1, T2, T2_for_D); // output is a struct
    om2 = docout_fit.Omega;
    c3 = docout_fit.C;
    d = docout_fit.D;

    // Updating Omega, Gamma
    omegaold = (om1 + om2) / 2;
    cold = (c1 + c2 + c3) / 3;

    // Update Dual variables
    u1 = u1 + rho * ( om1 - omegaold);
    u2 = u2 + rho * ( om2 - omegaold);
    u3 = u3 + rho * ( c3 - cold);
    u4 = u4 + rho * ( c1 - cold);
    u5 = u5 + rho * ( c2 - cold);
  }

  ADMM_block_out_clusterglasso ADMMblockout;
  ADMMblockout.om1 = om1;
  ADMMblockout.om2 = om2;
  ADMMblockout.c1 = c1;
  ADMMblockout.c2 = c2;
  ADMMblockout.c3 = c3;
  ADMMblockout.D = d;
  ADMMblockout.omega = omegaold;
  ADMMblockout.c = cold;
  ADMMblockout.u1 = u1;
  ADMMblockout.u2 = u2;
  ADMMblockout.u3 = u3;
  ADMMblockout.u4 = u4;
  ADMMblockout.u5 = u5;

  return(ADMMblockout);
}


// [[Rcpp::export]]
Rcpp::List LA_ADMM_clusterglasso_export(const int& it_out, const int& it_in, const arma::mat& S,
                                   const arma::mat& A, const arma::mat& Itilde, const arma::mat& A_for_C3, const arma::mat& A_for_T1, const arma::mat& T2, const arma::mat& T2_for_D,
                                   const double& lambda1, const double& lambda2, const double& rho, const bool& pendiag,
                                   const arma::mat& init_om, const arma::mat& init_u1, const arma::mat& init_u2,
                                   const arma::mat& init_c, const arma::mat& init_u3, const arma::mat& init_u4, const arma::mat& init_u5){
  // Input
  // it_out : scalar, T_stages of LA-ADMM algorithm
  // it_in : scalar, maximum number of iterations of (inner) ADMM algorithm
  // S : sample covariance matrix of dimension p times p
  // A : matrix of dimension p times p. In our case this is the identity matrix since we have (in the format of the old code) Omega = AC+D with A=I_pxp
  // Itilde : rbind(A, I_pxp)
  // A_for_C3 : matrix of dimension px(2p)
  // A_for_T1 : matrix of dimension dimension (p+p)x(p+p)
  // T2 : matrix of dimension (p+p) times p (requires Itilde)
  // T2_for_D : matrix of dimension p times p
  // Note that all these matrices can be computed at the start (now also in a more efficient way since A is the identity matrix; but I haven't changed this yet); we need them in the subproblem when we solve for Omega^(2), C^(3) and D
  // lambda1 : scalar, regularization parameter sparsity --> *IW: NOTE I CHANGED THIS COMPARED TO THE code.cpp document since on Dropbox in .pdf meetings file we use lambda1 for l1 norm and lambda2 for clustering penalty
  // lambda2: scalar, regularization parameter clustering --> *IW: NOTE I CHANGED THIS COMPARED TO THE code.cpp document since on Dropbox in .pdf meetings file we use lambda1 for l1 norm and lambda2 for clustering penalty
  // rho : scalar, parameter ADMM
  // pendiag : logical, penalize diagonal or not when solving for Omega^(1)
  // init_om : matrix of dimension p times p, initialization of Omega
  // init_u1 : matrix of dimension p times p, initialization of dual variable U1 of Omega^(1)
  // init_u2 : matrix of dimension p times p, initialization of dual variable U2 of Omega^(2)
  // init_c : matrix of dimension p times p, initialization of C
  // init_u3 : matrix of dimension p times p, initialization of dual variable U3 of C^(3)
  // init_u4 : matrix of dimension p times p, initialization of dual variable U4 of C^(1)
  // init_u5 : matrix of dimension p times p, initialization of dual variable U5 of C^(2)

  // Function : LA-ADMM updates

  // Preliminaries
  arma::mat in_om = init_om;
  arma::mat in_c = init_c;

  double rhoold = rho;
  double rhonew = rho;

  ADMM_block_out_clusterglasso ADMMout;

  for(int iout=0; iout < it_out; ++iout){

    ADMMout = ADMM_clusterglasso_block(S, A, Itilde, A_for_C3, A_for_T1, T2, T2_for_D, lambda1, lambda2, rhonew, pendiag, it_in,
                                  in_om, init_u1, init_u2, in_c, init_u3, init_u4, init_u5);
    in_om  = ADMMout.omega;
    in_c = ADMMout.c;
    rhonew = 2*rhoold;
    rhoold = rhonew;
  }

  // Remark: We don't need to return everything for final version, but can be useful now
  Rcpp::List results=Rcpp::List::create(
    Rcpp::Named("omega") = ADMMout.omega,
    Rcpp::Named("c") = ADMMout.c,
    Rcpp::Named("om1") = ADMMout.om1,
    Rcpp::Named("om2") = ADMMout.om2,
    Rcpp::Named("c1") = ADMMout.c1,
    Rcpp::Named("c2") = ADMMout.c2,
    Rcpp::Named("c3") = ADMMout.c3,
    Rcpp::Named("D") = ADMMout.D,
    Rcpp::Named("u1") = ADMMout.u1,
    Rcpp::Named("u2") = ADMMout.u2,
    Rcpp::Named("u3") = ADMMout.u3,
    Rcpp::Named("u4") = ADMMout.u4,
    Rcpp::Named("u5") = ADMMout.u5,
    Rcpp::Named("rho") = rhonew);

  return(results);
}
