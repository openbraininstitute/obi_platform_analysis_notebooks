#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;
#if defined(__cplusplus)
extern "C" {
#endif

extern void _CaDynamics_DC0_reg(void);
extern void _Ca_HVA2_reg(void);
extern void _Ca_LVAst_reg(void);
extern void _Ih_reg(void);
extern void _K_Pst_reg(void);
extern void _K_Tst_reg(void);
extern void _NaTg_reg(void);
extern void _Nap_Et2_reg(void);
extern void _ProbAMPANMDA_EMS_reg(void);
extern void _SK_E2_reg(void);
extern void _SKv3_1_reg(void);

void modl_reg() {
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");
    fprintf(stderr, " \"cadpyr/mechanisms//CaDynamics_DC0.mod\"");
    fprintf(stderr, " \"cadpyr/mechanisms//Ca_HVA2.mod\"");
    fprintf(stderr, " \"cadpyr/mechanisms//Ca_LVAst.mod\"");
    fprintf(stderr, " \"cadpyr/mechanisms//Ih.mod\"");
    fprintf(stderr, " \"cadpyr/mechanisms//K_Pst.mod\"");
    fprintf(stderr, " \"cadpyr/mechanisms//K_Tst.mod\"");
    fprintf(stderr, " \"cadpyr/mechanisms//NaTg.mod\"");
    fprintf(stderr, " \"cadpyr/mechanisms//Nap_Et2.mod\"");
    fprintf(stderr, " \"cadpyr/mechanisms//ProbAMPANMDA_EMS.mod\"");
    fprintf(stderr, " \"cadpyr/mechanisms//SK_E2.mod\"");
    fprintf(stderr, " \"cadpyr/mechanisms//SKv3_1.mod\"");
    fprintf(stderr, "\n");
  }
  _CaDynamics_DC0_reg();
  _Ca_HVA2_reg();
  _Ca_LVAst_reg();
  _Ih_reg();
  _K_Pst_reg();
  _K_Tst_reg();
  _NaTg_reg();
  _Nap_Et2_reg();
  _ProbAMPANMDA_EMS_reg();
  _SK_E2_reg();
  _SKv3_1_reg();
}

#if defined(__cplusplus)
}
#endif
