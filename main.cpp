#include <iostream>

#include "voxels.hpp"

#include <pybind11/embed.h>

void test_BF();
void get_k_on_tables(double **p_table, double **f_inv_table, double **ln_k0_on_table, double **kA_on_table,
					 double *ln_k0_off, double *kA_off, long *N_f, long *N_p);
void run_bruteforce_test(long * N_L, int PBC_mode, int accuracy_mode, int to_precomp_dist, int R_crit_mode,
						 double l_voxel, double R_0, double rho_init,
						 double rho_coex, double rho_l, double R_S0, double S0, double D_coef,
						 double dt, long Nt_max, int equil_mode, int rho_gen_mode,
						 int p_gen_mode, long N_active_sites, long p_high_ind, long NH_high,
						 long p_low_ind, long f_inv_ind,
						 long thermo_stride, long Rs_stride, long rho_stride,
						 double D_Hbig, double D_H, double S_thr, double ktot_thr,
						 double ln_k0_off, double kA_off,
						 double * ln_k0_on, double * kA_on,
						 double * p_arr, double * f_inv_arr,
						 long N_p, long N_f,
						 int verbose);

int main(int argc, char** argv) {

	test_BF();

	return 0;
}

void test_BF()
{
	double D_Hbig = 2.272;
	double D_H = 2.024;
	double rho_active_sites = 3000.0 / (3200.0 * 4600 * 800);
	double S_thr = 1.01;
	double ktot_thr = 0;

	int verbose = 1;

	long N_L[dim];
	N_L[0] = 10;
	N_L[1] = 10;
	N_L[2] = 10;
	double rho_coex = 3.85e-4;
	double rho_l = 1.57e-2;
	double S_init = 1.5;
	double R_0 = 10;
	double l_voxel = R_0 * 3;
	double Ncl_S0 = 50;
	double S0 = 1.5;
	double R_S0 = pow(3 * Ncl_S0 / (4 * M_PI * rho_l), 1.0 / 3.0);
	double D_coef = 0.2;
	double V = powi(l_voxel, dim) * prod_array(N_L, dim);
	long NH_high = 15;

//	long N_active_sites = lround(rho_active_sites * V);
//	int p_gen_mode = p_gen_mode_shuffle;
//	long N_active_sites = 1;
//	int p_gen_mode = p_gen_mode_1site;
	long N_active_sites = 2;
	int p_gen_mode = p_gen_mode_2neibrs;

	int equil_mode = equil_mode_no_eq;
	int rho_gen_mode = rho_gen_mode_RhoConst_noRs;
	int PBC_mode = PBC_mode_yes_PBC;
	int accuracy_mode = accuracy_mode_excl_nucls;
	int R_crit_mode = R_crit_mode_CNT;
	int to_precomp_dists = 1;

	long thermo_stride = 1;
	long Rs_stride = 2;
	long rho_stride = 5;

	long N_p, N_f;
	double *p_table, *f_inv_table;
	double ln_k0_off, kA_off;
	double *ln_k0_on, *kA_on;

	voxels::init_rand_C(0);

	get_k_on_tables(&p_table, &f_inv_table, &ln_k0_on, &kA_on, &ln_k0_off, &kA_off, (long*)&N_f, (long*)&N_p);

//	printf("%lf %lf %lf %lf\n", 1 / voxels::sqr(log(S_init)), )
//	double dt = 1.0 / (voxels::k_dens_fnc(ln_k0_off, kA_off, 1 / voxels::sqr(log(S_init))) * V * (rho_coex * S_init));
	double dt = 500.0;
	dt = -0.1;
	long Nt_max = 1000;

	run_bruteforce_test(N_L, PBC_mode, accuracy_mode, to_precomp_dists, R_crit_mode,
						 l_voxel, R_0, rho_coex * S_init,
						 rho_coex, rho_l, R_S0, S0, D_coef,
						 dt, Nt_max, equil_mode, rho_gen_mode,
						 p_gen_mode, N_active_sites, 7, NH_high,
						 0, 0,
						 thermo_stride, Rs_stride, rho_stride,
						 D_Hbig, D_H,
						 S_thr, ktot_thr,
						 ln_k0_off, kA_off,
						 ln_k0_on, kA_on,
						 p_table, f_inv_table, N_p, N_f,
						 verbose);

	c_free(& p_table);
	c_free(& f_inv_table);
	c_free(& ln_k0_on);
	c_free(& kA_on);
}

void run_bruteforce_test(long * N_L, int PBC_mode, int accuracy_mode, int to_precomp_dist, int R_crit_mode,
						 double l_voxel, double R_0, double rho_init,
						 double rho_coex, double rho_l, double R_S0, double S0, double D_coef,
						 double dt, long Nt_max, int equil_mode, int rho_gen_mode,
						 int p_gen_mode, long N_active_sites, long p_high_ind, long NH_high,
						 long p_low_ind, long f_inv_ind,
						 long thermo_stride, long Rs_stride, long rho_stride,
						 double D_Hbig, double D_H, double S_thr, double ktot_thr,
						 double ln_k0_off, double kA_off,
						 double * ln_k0_on, double * kA_on,
						 double * p_arr, double * f_inv_arr,
						 long N_p, long N_f,
						 int verbose)

{
// -------------- check input ----------------
	voxels::Run_parameters run_prms(dt, thermo_stride, Rs_stride,
									rho_stride,  Nt_max, equil_mode,
									S_thr, ktot_thr);

	voxels::System_parameters sys_prms((long*)N_L, PBC_mode, rho_coex, rho_l,
									   R_S0, S0, D_coef, l_voxel, R_0,
									   ln_k0_off, kA_off, to_precomp_dist, R_crit_mode,
									   accuracy_mode, verbose);


// ----------------- create return objects --------------
	long Nt = 0;

	voxels::Thermo_log thermo_log(thermo_stride);
	voxels::Rs_log rs_log(Rs_stride);
	voxels::Rho_log rho_log(rho_stride, sys_prms.N_vox);

	voxels::Proteins_state prot_init_state(sys_prms.N_L, rho_gen_mode, rho_init, dt < 0);

	voxels::Acetylation_state acetylation_state(sys_prms.N_L, N_active_sites,
												D_Hbig, D_H,
												sys_prms.N_vox, p_gen_mode,
												p_high_ind, NH_high, p_low_ind, f_inv_ind,
												ln_k0_on, kA_on, p_arr, f_inv_arr, N_p, N_f);
	acetylation_state.sort_by_indpos();

	voxels::Proteins_state prot_equild_state;
	voxels::get_equilibrated_state(prot_init_state, acetylation_state, sys_prms, run_prms,
								   & prot_equild_state);

	voxels::run_bruteforce_C(& Nt, prot_equild_state, acetylation_state, sys_prms, run_prms, thermo_log, rs_log, rho_log);

	thermo_log.print();
}

void get_k_on_tables(double **p_table, double **f_inv_table, double **ln_k0_on_table, double **kA_on_table,
					 double *ln_k0_off, double *kA_off, long *N_f, long *N_p)
{
	*N_p = 10;
	*N_f = 1;

	c_malloc(p_table, *N_p);
	c_malloc(f_inv_table, *N_f);
	c_malloc(ln_k0_on_table, *N_f * *N_p);
	c_malloc(kA_on_table, *N_f * *N_p);

	for(long i = 0; i < *N_p; ++i){
		(*p_table)[i] = (i + 1) / 10.0;
	}

	(*f_inv_table)[0] = 8;

	*kA_off = -1.11853838;
	double y0 = -0.7, y1 = -0.2, x0 = 0.1, xa = 0.1;
	for(long i_f = 0; i_f < *N_f; ++i_f){
		for(long i_p = 0; i_p < *N_p; ++i_p){
			(*kA_on_table)[i_p + i_f * *N_p] = y0 + (y1 - y0) * (1 - exp(-((*p_table)[i_p] - x0) / xa));
		}

	}
//	kA_on_table[1] = -0.70127648;
//	kA_on_table[2] = -0.08169488
//	kA_on_table[3] = -0.29949205
//	kA_on_table[4] = -0.35038911
//	kA_on_table[5] = -0.12901071
//	kA_on_table[6] = -0.17626831
//	kA_on_table[7] = -0.28891687
//	kA_on_table[8] = -0.20613492
//	kA_on_table[9] = -0.18230749
//	kA_on_table[10] = -0.20655942

	*ln_k0_off = -11.2293102;   // [if=0, ip=0]
	double ln_k0_fit2[3] = {1.47945428,  -3.58171513, -10.8015899};
	for(long i_f = 0; i_f < *N_f; ++i_f){
		for(long i_p = 0; i_p < *N_p; ++i_p){
			(*ln_k0_on_table)[i_p + i_f * *N_p] = ln_k0_fit2[2] + (*p_table)[i_p] * (ln_k0_fit2[1] + (*p_table)[i_p] * (ln_k0_fit2[0]));
		}
	}
//	ln_k0_on_table[1] = -11.20177293;   // [if=0, ip=1]
//	ln_k0_on_table[2] = -13.01194207;
//	ln_k0_on_table[3] = -11.67751134;
//	ln_k0_on_table[4] = -11.55261655;
//	ln_k0_on_table[5] = -12.81384969;
//	ln_k0_on_table[6] = -12.64709577;
//	ln_k0_on_table[7] = -12.17874434;
//	ln_k0_on_table[8] = -12.65178056;
//	ln_k0_on_table[9] = -12.92264834;
//	ln_k0_on_table[10] = -12.91465889;
}


