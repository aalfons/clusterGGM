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
                           const bool& pendiag, const arma::mat& W_sparsity){
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
      Omeganew(ir, ic) = softelem(soft_input(ir, ic), W_sparsity(ir, ic)*lambda/rho);
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

ADMM_block_out ADMM_taglasso_block(const arma::mat& S, const arma::mat& W_sparsity, const arma::mat& A, const arma::mat& Atilde, const arma::mat& A_for_gamma,
                                   const arma::mat& A_for_B, const arma::mat& C, const arma::mat& C_for_D, const double& lambda1,
                                   const double& lambda2, const double& rho, const bool& pendiag, const double& maxite,
                                   const arma::mat& init_om, const arma::mat& init_u1, const arma::mat& init_u3,
                                   const arma::mat& init_u4, const arma::mat& init_gam, const arma::mat& init_u2,
                                   const arma::mat& init_u5){
  // Input
  // S : sample covariance matrix of dimension p times p
  // W_sparsity : matrix of dimension p times p which contains the weights for the adaptive type of lasso sparsity penalty term
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
    om3 = solve_omega_soft(omegaold, u3, rho, lambda2, pendiag, W_sparsity); // output is a matrix of dimension p times p

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
Rcpp::List LA_ADMM_taglasso_export(const int& it_out, const int& it_in, const arma::mat& S, const arma::mat& W_sparsity,  const arma::mat& A,
                                   const arma::mat& Atilde, const arma::mat& A_for_gamma, const arma::mat& A_for_B,
                                   const arma::mat& C, const arma::mat& C_for_D, const double& lambda1,
                                   const double& lambda2, const double& rho, const bool& pendiag, const arma::mat& init_om,
                                   const arma::mat& init_u1, const arma::mat& init_u2, const arma::mat& init_u3,
                                   const arma::mat& init_gam, const arma::mat& init_u4, const arma::mat& init_u5){
  // Input
  // it_out : scalar, T_stages of LA-ADMM algorithm
  // it_in : scalar, maximum number of iterations of (inner) ADMM algorithm
  // S : sample covariance matrix of dimension p times p
  // W_sparsity : matrix of dimension p times p which contains the weights for the adaptive type of lasso sparsity penalty term
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

    ADMMout = ADMM_taglasso_block(S, W_sparsity, A, Atilde, A_for_gamma, A_for_B, C, C_for_D, lambda1, lambda2, rhonew, pendiag, it_in,
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
