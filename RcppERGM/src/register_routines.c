#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

SEXP _RcppERGM_EtaToMu_(SEXP gySEXP, SEXP freqSEXP, SEXP etaSEXP);
SEXP _RcppERGM_PEta_(SEXP gySEXP, SEXP freqSEXP, SEXP etaSEXP);
SEXP _RcppERGM_PND_(SEXP gySEXP, SEXP freqSEXP, SEXP etaSEXP);
SEXP _RcppERGM_MuToEta_(SEXP gySEXP, SEXP freqSEXP, SEXP targetSEXP, SEXP startSEXP, SEXP eps_fSEXP, SEXP eps_gSEXP, SEXP maxitSEXP);
SEXP _RcppERGM_EtaToMuNDM_(SEXP gySEXP, SEXP freqSEXP, SEXP etaSEXP);
SEXP _RcppERGM_PEtaNDM_(SEXP gySEXP, SEXP freqSEXP, SEXP etaSEXP);
SEXP _RcppERGM_PNDNDM_(SEXP gySEXP, SEXP freqSEXP, SEXP etaSEXP);
SEXP _RcppERGM_MuToEtaNDM_(SEXP gySEXP, SEXP freqSEXP, SEXP targetSEXP, SEXP startSEXP, SEXP eps_fSEXP, SEXP eps_gSEXP, SEXP maxitSEXP);

static const R_CallMethodDef CallEntries[] = {
    {"_RcppERGM_EtaToMu_", (DL_FUNC) &_RcppERGM_EtaToMu_, 3},
    {"_RcppERGM_PEta_", (DL_FUNC) &_RcppERGM_PEta_, 3},
    {"_RcppERGM_PND_", (DL_FUNC) &_RcppERGM_PND_, 3},
    {"_RcppERGM_MuToEta_", (DL_FUNC) &_RcppERGM_MuToEta_, 7},
    {"_RcppERGM_EtaToMuNDM_", (DL_FUNC) &_RcppERGM_EtaToMuNDM_, 3},
    {"_RcppERGM_PEtaNDM_", (DL_FUNC) &_RcppERGM_PEtaNDM_, 3},
    {"_RcppERGM_PNDNDM_", (DL_FUNC) &_RcppERGM_PNDNDM_, 3},
    {"_RcppERGM_MuToEtaNDM_", (DL_FUNC) &_RcppERGM_MuToEtaNDM_, 7},
    {NULL, NULL, 0}
};

void R_init_RcppERGM(DllInfo *info)
{
    R_registerRoutines(info, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(info, FALSE);
}
