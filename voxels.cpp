//
// Created by ypolyach on 10/27/21.
//

//#include <pybind11/pytypes.h>
//
//namespace py = pybind11;
//using namespace pybind11::literals;

#include "voxels.hpp"

py::int_ init_rand(long my_seed)
{
	voxels::init_rand_C(my_seed);
	return 0;
}

py::int_ get_seed()
{
	return voxels::seed;
}

py::int_ set_verbose(int new_verbose)
{
	voxels::verbose_default = new_verbose;
	return 0;
}

py::int_ get_verbose()
{
	return voxels::verbose_default;
}

py::tuple get_equil_modes()
{
	py::dict nums2strs;
	nums2strs[py::cast(equil_mode_no_eq)] = "no_eq";
	return py::make_tuple(py::dict("no_eq"_a=equil_mode_no_eq),
						  nums2strs);
}

py::tuple get_rho_gen_modes()
{
	py::dict nums2strs;
	nums2strs[py::cast(rho_gen_mode_RhoConst_noRs)] = "RhoConst_noRs";
	return py::make_tuple(py::dict("RhoConst_noRs"_a=rho_gen_mode_RhoConst_noRs),
						  nums2strs);
}

py::tuple get_accuracy_modes()
{
	py::dict nums2strs;
	nums2strs[py::cast(accuracy_mode_0)] = "0";
	nums2strs[py::cast(accuracy_mode_excl_nucls)] = "excl_nucls";
	nums2strs[py::cast(accuracy_mode_integr_outside)] = "integr_outside";
	return py::make_tuple(py::dict("0"_a=accuracy_mode_0,
								   "excl_nucls"_a=accuracy_mode_excl_nucls,
								   "integr_outside"_a=accuracy_mode_integr_outside),
						  nums2strs);
}

py::tuple get_Rcrit_modes()
{
	py::dict nums2strs;
	nums2strs[py::cast(R_crit_mode_const)] = "const";
	nums2strs[py::cast(R_crit_mode_CNT)] = "CNT";
	return py::make_tuple(py::dict("const"_a=R_crit_mode_const,
								   "CNT"_a=R_crit_mode_CNT),
						  nums2strs);
}

py::tuple get_p_gen_modes()
{
	py::dict nums2strs;
	nums2strs[py::cast(p_gen_mode_shuffle)] = "shuffle";
	nums2strs[py::cast(p_gen_mode_1site)] = "1site";
	nums2strs[py::cast(p_gen_mode_2neibrs)] = "2neibrs";
	return py::make_tuple(py::dict("shuffle"_a=p_gen_mode_shuffle,
								   "1site"_a=p_gen_mode_1site,
								   "2neibrs"_a=p_gen_mode_2neibrs),
						  nums2strs);
}

//void print_state(py_c_arr(int) state)
//{
//	py::buffer_info state_info = state.request();
//	int *state_ptr = static_cast<int *>(state_info.ptr);
//	assert(state_info.ndim == 1);
//
//	int L2 = state_info.shape[0];
//	int L = lround(sqrt(L2));
//	assert(L * L == L2);   // check for the full square
//
//	voxels::print_S(state_ptr, L, 0);
//}

py::tuple run_bruteforce(py_c_arr(double) rho_init_state, py_c_arr(double) Rs_init, py_c_arr(long) p_indpos_arr,
						 py_c_arr(int) droplet_on_inds_init,
						 double dt, long Nt_max,
						 long thermo_stride, long Rs_stride, long rho_stride,
						 py_c_arr(long) N_L_new, double l_voxel,
						 double ln_k0_off, double kA_off,
						 py_c_arr(double) ln_k0_on, py_c_arr(double) kA_on,
						 double R_0, double rho_coex, double rho_l, double R_S0, double S0, double D_coef,
						 py_c_arr(long) p_high_ind_arr, py_c_arr(long) NHhigh_arr,
						 py_c_arr(long) p_low_ind_arr, py_c_arr(long) f_inv_ind_arr,
						 py_c_arr(double) p_table, py_c_arr(double) f_inv_table,
						 double D_Hbig, double D_H, double S_thr, double ktot_thr,
						 int equil_mode, int to_precomp_dist, int R_crit_mode, int PBC_mode, int accuracy_mode,
						 std::optional<int> verbose_optional)

 ///
 /// \param rho_init_state : double[N_vox]; Initial densities in all voxels.
 /// \param Rs_init : double[N_vox]; Droplet sizes at t=0 for all voxels
 /// \param p_indpos_arr : lint[N_active_sites]; indpos-es of all sites with p>0
 /// \param droplet_on_inds_init : int[N_droplets]; on/off = 1/0 state of all existing droplets at t=0
 /// \param dt : double; timestep
 /// \param Nt_max : lint; The number of timesteps to run
 /// \param thermo_stride : lint; Save Thermo-log each this timesteps
 /// \param Rs_stride : lint; Save Rs-log each this timesteps
 /// \param rho_stride : lint; Save Rho-log each this timesteps
 /// \param N_L_new : lint[3]; Numbers of voxels in each dimension
 /// \param l_voxel : double; Voxel size (voxels are cubes for now)
 /// \param ln_k0_off, kA_off : double, double; k_off_dens(S) = exp(ln_k0_off + kA_off / ln(S)^2)
 /// \param ln_k0_on, kA_on : double[N_p * N_f], double[N_p * N_f]; k_off_dens(f, p, S) = exp(ln_k0_off(f, p) + kA_off(f, p) / ln(S)^2)
 /// \param R_0 : double; R_droplet(t=0) = R_0
 /// \param rho_coex : double; dilute-phase coexistance density
 /// \param rho_l : double; condense-phase coexistance density
 /// \param R_S0, S0 : double, double; R^*(S) = R_S0 * ln(S0) / ln(S)
 /// \param D_coef : double; Diffusion coefficient for dilute-phase proteins
 /// \param p_table, p_high_ind_arr : double[N_p], lint[N_active_sites]; p_high[i] = p_table[p_high_ind_arr[i]] for the i-th existing active site
 /// \param NHhigh_arr : lint[N_active_sites]; Number of high-p histones at each active site
 /// \param p_table, p_low_ind_arr : double[N_p], lint[N_active_sites]; p_low[i] = p_table[p_low_ind_arr[i]] for the i-th existing active site
 /// \param f_inv_table, f_inv_ind_arr : double[N_p], lint[N_active_sites]; f_inv[i] = f_inv_table[f_inv_ind_arr[i]] for the i-th existing active site
 /// \param D_Hbig : double; 1D box distance per histone, so N_histones = (1/f) * l_vox / D_Hbig
 /// \param D_H : double; V_histone = D_H^3 * pi/6
 /// \param S_thr : double; if >0, then the simulation stops when S < S_thr
 /// \param ktot_thr : double; if >0, then the simulation stops when {sum k_off + sum k_on < k_tot_thr}
 /// \param equil_mode : int; How to equilibrate the system before production
 /// \param to_precomp_dist : int; Whether to precomp voxel-voxel distances (RAM-heavy) or to compute them on the fly
 /// \param R_crit_mode : int; How to treat R^*(S)
 /// \param PBC_mode : int; Whether to use PBC
 /// \param accuracy_mode : int; What to include in the depletion intergrals
 /// \param verbose_optional : int; How verbose to be
 ///
 /// \return (...)
{
// -------------- check input ----------------
	int verbose = (verbose_optional.has_value() ? verbose_optional.value() : voxels::verbose_default);

	voxels::Run_parameters run_prms(dt, thermo_stride, Rs_stride,
									rho_stride,  Nt_max, equil_mode,
									S_thr, ktot_thr);

	voxels::System_parameters sys_prms(N_L_new, PBC_mode, rho_coex, rho_l,
									   R_S0, S0, D_coef, l_voxel, R_0,
									   ln_k0_off,kA_off, to_precomp_dist,
									   R_crit_mode, accuracy_mode, verbose);

// ----------------- create return objects --------------
	long Nt = 0;

	voxels::Thermo_log thermo_log(thermo_stride);
	voxels::Rs_log rs_log(Rs_stride);
	voxels::Rho_log rho_log(rho_stride, sys_prms.N_vox);

	voxels::Proteins_state prot_init_state(sys_prms.N_L,
										   extract_arr_from_pybind11(rho_init_state, sys_prms.N_vox, 1),
										   extract_arr_from_pybind11(Rs_init, 0, 1),
										   extract_arr_from_pybind11(droplet_on_inds_init, 0, 1),
										   dt < 0,
										   1);

	long N_active_sites = p_indpos_arr.shape()[0];
	voxels::Acetylation_state acetylation_state(N_active_sites, D_Hbig, D_H,
												sys_prms.N_vox, p_indpos_arr, NHhigh_arr,
												p_high_ind_arr, p_low_ind_arr, f_inv_ind_arr,
												ln_k0_on, kA_on, p_table, f_inv_table);
	acetylation_state.sort_by_indpos();

	voxels::Proteins_state prot_equild_state;
	voxels::get_equilibrated_state(prot_init_state, acetylation_state, sys_prms, run_prms,
								   & prot_equild_state);
	voxels::run_bruteforce_C(& Nt, prot_equild_state, acetylation_state, sys_prms, run_prms, thermo_log, rs_log, rho_log);

//	thermo_log.print();

//	int N_last_elements_to_print = std::min(Nt_OP_saved, (long)10);

	py_c_arr(double) S;
	py_c_arr(double) rate_off_total;
	py_c_arr(double) rate_on_total;
	py_c_arr(long) N_droplets;
	py_c_arr(long) N_on_droplets;
	py_c_arr(long) N_merges;
	py_c_arr(long) timestep;
	py_c_arr(double) time;
	py_c_arr(double) Va;
	py_c_arr(double) Vb;
	py_c_arr(double) Vc;
	if(thermo_stride > 0){
		c_arr_to_py_arr(& S, & thermo_log.S, thermo_log.len);
		c_arr_to_py_arr(& rate_off_total, & thermo_log.rate_off_total, thermo_log.len);
		c_arr_to_py_arr(& rate_on_total, & thermo_log.rate_on_total, thermo_log.len);
		c_arr_to_py_arr(& N_droplets, & thermo_log.N_droplets, thermo_log.len);
		c_arr_to_py_arr(& N_on_droplets, & thermo_log.N_on_droplets, thermo_log.len);
		c_arr_to_py_arr(& N_merges, & thermo_log.N_merges, thermo_log.len);
		c_arr_to_py_arr(& timestep, & thermo_log.timestep, thermo_log.len);
		c_arr_to_py_arr(& time, & thermo_log.time, thermo_log.len);
		c_arr_to_py_arr(& Va, & thermo_log.Va, thermo_log.len);
		c_arr_to_py_arr(& Vb, & thermo_log.Vb, thermo_log.len);
		c_arr_to_py_arr(& Vc, & thermo_log.Vc, thermo_log.len);
	}

	py_c_arr(double) Rs;
	py_c_arr(long) Rs_indpos;
	py_c_arr(int) Rs_is_on;
	py_c_arr(long) N_Rs;
	py_c_arr(double) Rs_time;
	if(Rs_stride > 0){
		long N_Rs_total = sum_array(rs_log.N_Rs, rs_log.len);
		c_arr_to_py_arr(& N_Rs, & rs_log.N_Rs, rs_log.len);
		c_arr_to_py_arr(& Rs_time, & rs_log.time, rs_log.len);
		if(N_Rs_total > 0){
			c_arr_to_py_arr(& Rs, & rs_log.Rs, N_Rs_total);
			c_arr_to_py_arr(& Rs_indpos, & rs_log.indpos, N_Rs_total);
			c_arr_to_py_arr(& Rs_is_on, & rs_log.is_on, N_Rs_total);
		}
	}

	py_c_arr(double) rho_states;
	py_c_arr(double) rho_time;
	if(rho_stride > 0){
		c_arr_to_py_arr(& rho_states, & rho_log.states, rho_log.len * rho_log.state_size);
		c_arr_to_py_arr(& rho_time, & rho_log.time, rho_log.len);
	}

	return py::make_tuple(py::dict("S"_a = S,
								   "rate_off_total"_a = rate_off_total,
								   "rate_on_total"_a = rate_on_total,
								   "N_droplets"_a = N_droplets,
								   "N_on_droplets"_a = N_on_droplets,
								   "N_merges"_a = N_merges,
								   "Va"_a = Va,
								   "Vb"_a = Vb,
								   "Vc"_a = Vc,
								   "time"_a = time,
								   "timestep"_a = timestep),
						  py::dict("Rs_flatten"_a = Rs,
								   "Rs_indpos_flatten"_a = Rs_indpos,
								   "Rs_is_on_flatten"_a = Rs_is_on,
								   "N_Rs"_a = N_Rs,
								   "time"_a = Rs_time),
						  py::dict("rho_states_flatten"_a = rho_states,
								   "time"_a = rho_time,
								   "N_frames"_a = rho_log.len)
						  );
}


namespace voxels
{
//	std::mt19937 *gen_mt19937;
    gsl_rng *rng;
    long seed;
    int verbose_default;

	double k_dens_fnc(double ln_k0, double kA, double x){ return exp(ln_k0 + x * kA); }

	void indpos2pos(long indpos, const long *N_L, long *pos){
		long p = 1;
		for(long i = dim - 1; i >= 0; --i){
			pos[i] = (indpos / p) % N_L[i];
			p *= N_L[i];
		}
	}
	void indpos2pos(long indpos, long *N_L, double *pos, double l){
		long pos_int[dim];
		indpos2pos(indpos, N_L, pos_int);
		FOR_DIM pos[_i_dim] = (pos_int[_i_dim] + 0.5) * l;
	}
	long pos2indpos(const long *pos, const long *N_L){
		long _indpos = pos[0];
		for(int i = 1; i < dim; ++i)
			_indpos = _indpos * N_L[i] + pos[i];
		return _indpos;
	}
	long pos2indpos(const double *pos, const long *N_L, double l){
		long _indpos = long(pos[0] / l);
		for(int i = 1; i < dim; ++i)
			_indpos = _indpos * N_L[i] + long(pos[i] / l);
		return _indpos;
	}
	long pos2indpos(long x, long y, long z, const long *N_L){
		long pos[dim] = {x, y, z};
		return pos2indpos(pos, N_L);
	}

	void Rho_log::clear()
	{
		c_free(& this->states);
		c_free(& this->time);
		Log::clear_lens();
	}

	void Rho_log::alloc(long max_len_new)
	{
		c_malloc(& this->states, max_len_new * this->state_size);
		c_malloc(& this->time, max_len_new);
		this->max_len = max_len_new;
	}

	void Rho_log::alloc() { this->alloc(this->max_len); }

	void Rho_log::resize(long max_len_new)
	{
		c_realloc_this(& this->states, max_len_new * this->state_size);
		c_realloc_this(& this->time, max_len_new);
		this->max_len = max_len_new;
	}

	void Rho_log::add_record(double time_new, double *state_new, int verbose)
	{
		if(this->len + 1 > this->max_len){
			this->resize(std::max(this->len + 1, this->max_len * 2));
			if(verbose >= 2){
				printf("Rho_log: realloced to %ld\n", this->max_len);
			}
		}

		memcpy(&(this->states[this->state_size * this->len]), state_new, sizeof(double) * state_size);
		this->time[this->len] = time_new;

		++ this->len;
	}

	void Rho_log::print_params()
	{
		printf("Rho-log: ");
		Log::print_params();
		printf("state_size = %ld\n", this->state_size);
	}

	void Rs_log::clear()
	{
		c_free(& this->Rs);
		c_free(& this->indpos);
		c_free(& this->is_on);
		c_free(& this->N_Rs);
		c_free(& this->time);
		Log::clear_lens();
	}

	void Rs_log::alloc(long max_len_new, long max_N_total_new)
	{
		c_malloc(& this->N_Rs, max_len_new);
		c_malloc(& this->time, max_len_new);

		c_malloc(& this->Rs, max_N_total_new);
		c_malloc(& this->indpos, max_N_total_new);
		c_malloc(& this->is_on, max_N_total_new);

		this->max_len = max_len_new;
		this->max_N_total = max_N_total_new;
	}

	void Rs_log::alloc() { this->alloc(this->max_len, this->max_N_total); }

	void Rs_log::resize_Rs(long max_N_total_new)
	{
		c_realloc_this(& this->Rs, max_N_total_new);
		c_realloc_this(& this->indpos, max_N_total_new);
		c_realloc_this(& this->is_on, max_N_total_new);

		this->max_N_total = max_N_total_new;
	}

	void Rs_log::resize_NR(long max_len_new)
	{
		c_realloc_this(& this->N_Rs, max_len_new);
		c_realloc_this(& this->time, max_len_new);

		this->max_len = max_len_new;
	}

	void Rs_log::add_record(double time_new, long N_R_new, const double *Rs_new, const long *indpos_new, const int *is_on_new, int verbose)
	{
		if(this->len + 1 > this->max_len){
			this->resize_NR(std::max(this->len + 1, this->max_len * 2));
			if(verbose >= 2){
				printf("Rs_log-NR: realloced to %ld\n", this->max_len);
			}

		}
		if(this->N_total + N_R_new > this->max_N_total){
			this->resize_Rs( std::max(this->N_total + N_R_new, this->max_N_total * 2));
			if(verbose >= 2){
				printf("Rs_log-Rs: realloced to %ld\n", this->max_N_total);
			}
		}

		this->N_Rs[this->len] = N_R_new;
		this->time[this->len] = time_new;

		memcpy(&(this->indpos[this->N_total]), indpos_new, sizeof(long) * N_R_new);
		memcpy(&(this->is_on[this->N_total]), is_on_new, sizeof(int) * N_R_new);
		for(long i = 0; i < N_R_new; ++i){
			this->Rs[this->N_total + i] = Rs_new[indpos_new[i]];
		}

		++ this->len;
		this->N_total += N_R_new;
	}

	void Rs_log::print_params()
	{
		printf("Rs-log: ");
		Log::print_params();
		printf("N_total = %ld, max_N_total = %ld\n", this->N_total, this->max_N_total);
	}

	void Thermo_log::clear()
	{
		c_free(& this->S);
		c_free(& this->N_droplets);
		c_free(& this->N_on_droplets);
		c_free(& this->N_merges);
		c_free(& this->timestep);
		c_free(& this->time);
		c_free(& this->rate_off_total);
		c_free(& this->rate_on_total);
		c_free(& this->Va);
		c_free(& this->Vb);
		c_free(& this->Vc);
		Log::clear_lens();
	}

	void Thermo_log::alloc(long max_len_new)
	{
		c_malloc(& this->S, max_len_new);
		c_malloc(& this->N_droplets, max_len_new);
		c_malloc(& this->N_on_droplets, max_len_new);
		c_malloc(& this->N_merges, max_len_new);
		c_malloc(& this->timestep, max_len_new);
		c_malloc(& this->time, max_len_new);
		c_malloc(& this->rate_off_total, max_len_new);
		c_malloc(& this->rate_on_total, max_len_new);
		c_malloc(& this->Va, max_len_new);
		c_malloc(& this->Vb, max_len_new);
		c_malloc(& this->Vc, max_len_new);
		this->max_len = max_len_new;
	}

	void Thermo_log::resize(long max_len_new)
	{
		c_realloc_this(& this->S, max_len_new);
		c_realloc_this(& this->N_droplets, max_len_new);
		c_realloc_this(& this->N_on_droplets, max_len_new);
		c_realloc_this(& this->N_merges, max_len_new);
		c_realloc_this(& this->timestep, max_len_new);
		c_realloc_this(& this->time, max_len_new);
		c_realloc_this(& this->rate_off_total, max_len_new);
		c_realloc_this(& this->rate_on_total, max_len_new);
		c_realloc_this(& this->Va, max_len_new);
		c_realloc_this(& this->Vb, max_len_new);
		c_realloc_this(& this->Vc, max_len_new);
		this->max_len = max_len_new;
	}

	Thermo_log::~Thermo_log(){
		this->clear();
	}


	void Thermo_log::add_record(double S_new, long N_droplets_new, long N_on_droplets_new, long N_merges_new,
								long timestep_new, double time_new, double rate_off_total_new, double rate_on_total_new,
								double Va_new, double Vb_new, double Vc_new, int verbose)
	{
		if(this->len + 1 > this->max_len){
			this->resize(std::max(this->len + 1, this->max_len * 2));
			if(verbose >= 2){
				printf("Thermo_log: realloced to %ld\n", this->max_len);
			}
		}

		this->S[this->len] = S_new;
		this->N_droplets[this->len] = N_droplets_new;
		this->N_on_droplets[this->len] = N_on_droplets_new;
		this->N_merges[this->len] = N_merges_new;
		this->timestep[this->len] = timestep_new;
		this->time[this->len] = time_new;
		this->rate_off_total[this->len] = rate_off_total_new;
		this->rate_on_total[this->len] = rate_on_total_new;
		this->Va[this->len] = Va_new;
		this->Vb[this->len] = Vb_new;
		this->Vc[this->len] = Vc_new;

		++this->len;
	}

	void Thermo_log::print_params()
	{
		printf("Thermo-log: ");
		Log::print_params();
	}

	void Thermo_log::print()
	{
		printf("%10s %10s %10s %10s %10s %10s %10s %10s\n",
			   "Timestep", "Time", "Supersat", "N_droplets", "N_on_drop", "N_merges",
			   "k_tot_off", "k_tot_on");
		for(long i = 0; i < this->len; ++i){
			printf("%10ld %10lf %10lf %10ld %10ld %10ld %10e %10e %10lf %10lf %10lf\n",
				   this->timestep[i], this->time[i], this->S[i], this->N_droplets[i], this->N_on_droplets[i], this->N_merges[i],
				   this->rate_off_total[i], this->rate_on_total[i], this->Va[i], this->Vb[i], this->Vc[i]);
		}
	}

	void Log::print_params()
	{
		printf("stride = %ld, len = %ld, max_len = %ld\n", this->stride, this->len, this->max_len);
	}


	Proteins_state::Proteins_state(const long* N_L_new, long N_droplets_new, int to_comp_dots_new, int to_alloc):
			to_comp_dots(to_comp_dots_new), alloced_Rs(0), alloced_rho(0)
	{
		this->change_state_size(N_L_new);
		this->N_droplets = N_droplets_new;
		this->N_droplet_max = N_droplets_new;
		if(to_alloc){
			this->alloc();
		}
	}

	Proteins_state::Proteins_state(py_c_arr(long) & N_L_new, long N_droplets_new, int to_comp_dots_new, int to_alloc):
			Proteins_state(extract_arr_from_pybind11(N_L_new, dim, 1), N_droplets_new, to_comp_dots_new, to_alloc)
	{}

	Proteins_state::Proteins_state(const long *N_L_new, int rho_gen_mode, double rho_init, int to_comp_dots_new):
		to_comp_dots(to_comp_dots_new)
	{
		this->change_state_size(N_L_new);
		this->alloc();
		this->generate(rho_gen_mode, rho_init);
	}

	Proteins_state::~Proteins_state(){
		if(this->alloced_rho){
			this->clear();
		}
		if(this->alloced_Rs){
			this->clear_Rs();
		}
	}

	Proteins_state::Proteins_state(const long* N_L_new, double *rho_state_new, double* Rs_new,
								   const int *droplet_on_inds_new, int to_comp_dots_new, int to_alloc):
								   to_comp_dots(to_comp_dots_new), alloced_Rs(0), alloced_rho(0)
	{
		this->change_state_size(N_L_new);
		this->N_droplets = 0;
		for(long i = 0; i < this->N_vox; ++i){
			if(Rs_new[i] > 0){
				++ this->N_droplets;
			}
		}
		this->N_droplet_max = this->N_droplets;

		if(to_alloc){
			this->alloc();
			memcpy(this->rho_state, rho_state_new, sizeof(double) * this->N_vox);
			memcpy(this->Rs, Rs_new, sizeof(double) * this->N_vox);
		} else {
			this->alloc_Rs();
			this->rho_state = rho_state_new;
			this->Rs = Rs_new;
		}

		long j = 0;
		for(long i = 0; i < this->N_vox; ++i){
			if(this->Rs[i] > 0){
				this->Rs_indpos[j] = i;
				this->droplet_on_inds[j] = droplet_on_inds_new[j];
				++j;
			}
		}

		this->update_R_powers();
	}


	void Proteins_state::copy_from(voxels::Proteins_state * src, int to_alloc)
	{
		this->change_state_size(src->N_L);
		this->N_droplets = src->N_droplets;
		this->N_droplet_max = src->N_droplet_max;
		this->to_comp_dots = src->to_comp_dots;
		this->rho_inf = src->rho_inf;
		this->rho_inf_dot = src->rho_inf_dot;
		this->k_on_total = src->k_on_total;
		this->k_off_total = src->k_off_total;
		this->Va = src->Va;
		this->Vb = src->Vb;
		this->Vb0 = src->Vb0;
		this->Vc = src->Vc;
		this->Va_dot = src->Va_dot;
		this->Vb_dot = src->Vb_dot;
		this->Vb0_dot = src->Vb0_dot;
		this->Vc_dot = src->Vc_dot;

		if(to_alloc){
			this->alloc();
			memcpy(this->rho_state, src->rho_state, sizeof(double) * this->N_vox);
			memcpy(this->Rs, src->Rs, sizeof(double) * this->N_vox);
			if(src->rates_on) memcpy(this->rates_on, src->rates_on, sizeof(double) * this->N_vox);
			if(src->rates_off) memcpy(this->rates_off, src->rates_off, sizeof(double) * this->N_vox);
			if(src->invlogS_state) memcpy(this->invlogS_state, src->invlogS_state, sizeof(double) * this->N_vox);
			if(this->N_droplet_max > 0){
				memcpy(this->Rs_indpos, src->Rs_indpos, sizeof(long) * this->N_droplet_max);
				memcpy(this->droplet_on_inds, src->droplet_on_inds, sizeof(int) * this->N_droplet_max);
				memcpy(this->R2, src->R2, sizeof(double) * this->N_droplet_max);
				memcpy(this->R3, src->R3, sizeof(double) * this->N_droplet_max);
				if(src->Rs_dot) memcpy(this->Rs_dot, src->Rs_dot, sizeof(double) * this->N_droplet_max);
				if(src->dR) memcpy(this->dR, src->dR, sizeof(double) * this->N_droplet_max);
				memcpy(this->Sj, src->Sj, sizeof(double) * this->N_droplet_max);
				if(src->to_comp_dots) memcpy(this->Sj_dot, src->Sj_dot, sizeof(double) * this->N_droplet_max);
			}
		} else {
			this->rho_state = src->rho_state;
			this->Rs = src->Rs;
			this->rates_on = src->rates_on;
			this->rates_off = src->rates_off;
			this->invlogS_state = src->invlogS_state;
			this->Rs_indpos = src->Rs_indpos;
			this->droplet_on_inds = src->droplet_on_inds;
			this->R2 = src->R2;
			this->R3 = src->R3;
			this->dR = src->dR;
			this->Sj = src->Sj;
			this->Sj_dot = src->Sj_dot;
		}
	}

	void Proteins_state::generate(int rho_gen_mode, double rho_init)
	{
		if(rho_gen_mode == rho_gen_mode_RhoConst_noRs){
			for(long i = 0; i < this->N_vox; ++i){
				this->rho_state[i] = rho_init;
				this->Rs[i] = 0;
			}
		}
	}

	void Proteins_state::change_state_size(const long* N_L_new)
	{
//		printf("4300\n"); STP
		memcpy(this->N_L, N_L_new, sizeof(long) * dim);
//		printf("4400\n"); STP
		this->N_vox = prod_array(this->N_L, dim);
//		printf("4500\n"); STP
	}

	void Proteins_state::clear()
	{
		if((this->N_vox > 0) && (this->alloced_rho)){
			c_free(& this->rho_state);
			c_free(& this->rates_on);
			c_free(& this->rates_off);
			c_free(& this->Rs);
			c_free(& this->invlogS_state);
			this->N_vox = 0;
		}

		this->clear_Rs();
	}

	void Proteins_state::clear_Rs()
	{
		if((this->N_droplet_max > 0) && (this->alloced_Rs)){
			c_free(& this->Rs_indpos);
			c_free(& this->droplet_on_inds);
			c_free(& this->R2);
			c_free(& this->R3);
			c_free(& this->Rs_dot);
			c_free(& this->dR);
			c_free(& this->Sj);
			if(this->to_comp_dots){
				c_free(& this->Sj_dot);
			}

			N_droplets = 0;
			N_droplet_max = 0;
		}
	}

	void Proteins_state::alloc()
	{
		if(this->N_vox > 0){
			c_malloc(& this->Rs, this->N_vox);
			c_malloc(& this->rho_state, this->N_vox);
			c_malloc(& this->rates_on, this->N_vox);
			c_malloc(& this->rates_off, this->N_vox);
			c_malloc(& this->invlogS_state, this->N_vox);

			this->alloced_rho = 1;
		}

		this->alloc_Rs();
	}

	void Proteins_state::alloc_Rs()
	{
		if(this->N_droplet_max > 0){
			c_malloc(& this->Rs_indpos, this->N_droplet_max);
			c_malloc(& this->droplet_on_inds, this->N_droplet_max);
			c_malloc(& this->R2, this->N_droplet_max);
			c_malloc(& this->R3, this->N_droplet_max);
			c_malloc(& this->Rs_dot, this->N_droplet_max);
			c_malloc(& this->dR, this->N_droplet_max);
			c_malloc(& this->Sj, this->N_droplet_max);
			if(this->to_comp_dots){
				c_malloc(& this->Sj_dot, this->N_droplet_max);
			}

			this->alloced_Rs = 1;
		}
	}

	void Proteins_state::realloc_Rs(long len_new)
	{
		if(len_new > 0){
			c_realloc_this(& this->Rs_indpos, len_new);
			c_realloc_this(& this->droplet_on_inds, len_new);
			c_realloc_this(& this->R2, len_new);
			c_realloc_this(& this->R3, len_new);
			c_realloc_this(& this->Rs_dot, len_new);
			c_realloc_this(& this->dR, len_new);
			c_realloc_this(& this->Sj, len_new);
			if(this->to_comp_dots){
				c_realloc_this(& this->Sj_dot, len_new);
			}

			this->alloced_Rs = 1;
		} else {
			this->clear_Rs();
		}

		this->N_droplet_max = len_new;
	}

	void Proteins_state::add_droplet(double R, long indpos, int is_on)
	{
		if(this->N_droplets + 1 > this->N_droplet_max){
			this->realloc_Rs(std::max(this->N_droplet_max * 2, this->N_droplets + 1));
		}

		this->Rs_indpos[this->N_droplets] = indpos;
		this->Rs[indpos] = R;
		this->droplet_on_inds[this->N_droplets] = is_on;
		this->R2[this->N_droplets] = R * R;
		this->R3[this->N_droplets] = this->R2[this->N_droplets] * R;

		++ this->N_droplets;
	}

	void Proteins_state::print_params()
	{
		printf("N_L = (%ld, %ld, %ld), N_vox = %ld, N_droplets = %ld\n", this->N_L[0], this->N_L[1], this->N_L[2], this->N_vox, this->N_droplets);
	}

	double Proteins_state::protein_total_amount() const
	{
		return sum_array(this->rho_state, this->N_vox);
	}

	double Proteins_state::protein_total_amount(double l) const
	{
		return this->protein_total_amount() * powi(l, dim);
	}

	void Proteins_state::update_R_powers()
	{
		double r, r2, r3;
		for(long i = 0; i < this->N_droplets; ++i){
			r = this->Rs[this->Rs_indpos[i]];
			r2 = r * r;
			r3 = r2 * r;
			this->R2[i] = r2;
			this->R3[i] = r3;
		}
	}

	struct sgminf_fnc_params{
		double b, c;
	};

	double sgminf_fnc(double x, void *params)
	{
		struct sgminf_fnc_params *p	= (struct sgminf_fnc_params *) params;

		double b = p->b;
		double c = p->c;

		return c / (b - 1 / log(1+x)) - x;
	}

	double sgminf_fnc_derv (double x, void *params)
	{
		struct sgminf_fnc_params *p	= (struct sgminf_fnc_params *) params;

		double b = p->b;
		double c = p->c;

		double _b = b * log(1+x) - 1;

		return -c/((1+x) * _b * _b) - 1;
	}

	void sgminf_fnc_fdf (double x, void *params, double *y, double *dy)
	{
		struct sgminf_fnc_params *p	= (struct sgminf_fnc_params *) params;

		double b = p->b;
		double c = p->c;
		double _b = b * log(1+x);

		*y = c / (b * (1 - 1 / _b)) - x;
		_b -= 1;
		*dy = -c/((1+x) * _b * _b) - 1;
	}

	double sgminf_solve(double b, double c, int verbose){
		const gsl_root_fdfsolver_type *T;
		gsl_root_fdfsolver *s;
		gsl_function_fdf FDF;
		struct sgminf_fnc_params sgminf_params{b, c};
//		double x = (fmax(1.0, exp(1/sgminf_params.b) - 1.0)) * (sgminf_params.c > 0 ? 2 - exp(-sgminf_params.c) : 0.99);
		double x = (exp(1/sgminf_params.b) - 1.0) * (sgminf_params.c > 0 ? 2 - exp(-sgminf_params.c) : 0.99);
//		double x = (sgminf_params.b > 0 ? exp(1/sgminf_params.b) - 1.0 : 1) * (sgminf_params.c > 0 ? 2 - exp(-sgminf_params.c) : (sgminf_params.c >= -1 ? 0.99 : -2*(sgminf_params.c + 1)));

		FDF.f = &sgminf_fnc;
		FDF.df = &sgminf_fnc_derv;
		FDF.fdf = &sgminf_fnc_fdf;
		FDF.params = &sgminf_params;

		T = gsl_root_fdfsolver_newton;
		s = gsl_root_fdfsolver_alloc (T);
		gsl_root_fdfsolver_set (s, &FDF, x);

		if(verbose > 2){
			printf ("using %s method\n", gsl_root_fdfsolver_name (s));
		}

		double x_prev;
		double epsabs = 0;
		double epsrel = 1e-3;
		int status;
		long iter = 0, max_iter = 100;
		do
		{
			iter++;
//			status = gsl_root_fdfsolver_iterate (s);
			gsl_root_fdfsolver_iterate (s);
			x_prev = x;
			x = gsl_root_fdfsolver_root (s);
			status = gsl_root_test_delta (x, x_prev, epsabs, epsrel);
		}
		while ((status == GSL_CONTINUE) && (iter < max_iter));

		if (status == GSL_SUCCESS){
			if(verbose > 2){
				printf ("Converged, epsabs = %lf, epsrel = %lf:\n", epsabs, epsrel);
			}
		} else {
			if(verbose > 2){
				printf ("Reached max-iter = %ld, epsabs = %lf, epsrel = %lf\n", max_iter, epsabs, epsrel);
			}
		}
		gsl_root_fdfsolver_free (s);

		return x;
	}

	void Proteins_state::update_rho_inf(voxels::System_parameters & sys_prms)
	{
		if(this->N_droplets == 0){
			this->rho_inf = sys_prms.protein_total_amount / sys_prms.V;
		} else {
			this->update_Va();
			this->update_Sj(sys_prms);

			switch (sys_prms.R_crit_mode) {
				case R_crit_mode_const:   // R_crit = const
					this->update_Vb(sys_prms);
					this->rho_inf = (sys_prms.protein_total_amount + this->Vb * sys_prms.rho_coex - this->Va * sys_prms.rho_l) / (sys_prms.V + this->Vb - this->Va);
					break;
				case R_crit_mode_CNT:   // R_crit ~ 1/ln(S)
					this->update_Vbs(sys_prms);

					double b = (sys_prms.V - this->Va + this->Vb0) / this->Vc;
					double c = (sys_prms.protein_total_amount / sys_prms.rho_coex - sys_prms.V - this->Va * (sys_prms.rho_l / sys_prms.rho_coex - 1)) / this->Vc;

					if((b <= 0) || (c <= 0)){
						printf("b = %e, c = %e\n", b, c);
						throw std::runtime_error("b,c <= 0 occurred, but it is not supported for now. Please use R_crit_mode=const for now\n");
					}

					double sgm_inf = sgminf_solve(b, c, sys_prms.verbose);

					this->rho_inf = sys_prms.rho_coex * (1 + sgm_inf);

					break;
			}
		}
	}

	void Proteins_state::update_logS(double rho_coex, double rho_thr)
	{
		for(long i = 0; i < this->N_vox; ++i){
			if(this->rho_state[i] > rho_thr){
				this->invlogS_state[i] = 1 / log(this->rho_state[i] / rho_coex);
			} else {
				this->invlogS_state[i] = 0;
			}
		}
	}

	void Proteins_state::update_dR(voxels::System_parameters &sys_prms)
	{
		long indpos;
		double R_crit_0 = sys_prms.R_crit_mode == R_crit_mode_const ? sys_prms.R_S0 / log(sys_prms.S0) :
						  									sys_prms.R_S0 / log(this->rho_inf / sys_prms.rho_coex);
		for(long i = 0; i < this->N_droplets; ++i){
			indpos = this->Rs_indpos[i];
//			this->dR[i] = this->Rs[indpos] - sys_prms.R_S0 * this->invlogS_state[indpos];
			this->dR[i] = this->Rs[indpos] - R_crit_0;
		}
	}

	void Proteins_state::update_Sj(voxels::System_parameters &sys_prms)
	{
		for(long i = 0; i < this->N_droplets; ++i){
			this->Sj[i] = this->comp_Sj(sys_prms, i);
		}
	}

	void Proteins_state::update_Rdot(voxels::System_parameters & sys_prms)
	{
		for(long i = 0; i < this->N_droplets; ++i){
			this->Rs_dot[i] = (sys_prms.D / sys_prms.rho_l) * (this->rho_inf - sys_prms.rho_coex) * this->dR[i] / sqr(this->Rs[this->Rs_indpos[i]]);
		}
	}

	void Proteins_state::update_rates(voxels::System_parameters & sys_prms, Acetylation_state & acetyl_state)
	{
		double N_chromatin, N_chr_onoff_ratio;
		long indpos_i = 0;
		long i_p, i_f, i_k;

		double v_off = 0;
		double n_sites_on = 0;
		double f_inv = 0;
		double invlogS2;
		if(acetyl_state.N_active_sites > 0){
			f_inv = acetyl_state.f_inv_table[acetyl_state.f_inv_ind_arr[0]];
			N_chromatin = sys_prms.l_voxel / acetyl_state.D_Hbig * f_inv;
			v_off = sys_prms.v_voxel - powi(acetyl_state.D_H, 3) * (M_PI / 6) * N_chromatin;
			N_chr_onoff_ratio = std::min(1.0, acetyl_state.NHhigh_arr[0] / N_chromatin);
			n_sites_on = 8 * sys_prms.l_voxel / (acetyl_state.D_Hbig) * sqrt(f_inv);
		}

		for(long i = 0; i < this->N_vox; ++i){
			if(this->rho_state[i] <= sys_prms.rho_coex){
				this->rates_off[i] = 0;
				this->rates_on[i] = 0;
//				if(acetyl_state.indpos_arr[indpos_i] == i) ++ indpos_i;
			} else {
				if(acetyl_state.N_active_sites <= indpos_i){   // all active sites analyzed
					this->rates_on[i] = 0;
					this->rates_off[i] = k_dens_fnc(sys_prms.ln_k0_off, sys_prms.kA_off, sqr(this->invlogS_state[i])) * sys_prms.v_voxel * this->rho_state[i];
				} else {
					if(acetyl_state.indpos_arr[indpos_i] == i){   // indpos_arr has to be ordered increasingly
						invlogS2 = sqr(this->invlogS_state[i]);
						f_inv = acetyl_state.f_inv_table[acetyl_state.f_inv_ind_arr[indpos_i]];
						N_chromatin = sys_prms.l_voxel / acetyl_state.D_Hbig * f_inv;
						v_off = sys_prms.v_voxel - powi(acetyl_state.D_H, 3) * (M_PI / 6.0) * N_chromatin;
						n_sites_on = 8 * sys_prms.l_voxel / (acetyl_state.D_Hbig) * sqrt(f_inv);
						i_p = acetyl_state.p_high_ind_arr[indpos_i];
						i_f = acetyl_state.f_inv_ind_arr[indpos_i];
						i_k = i_f * acetyl_state.N_p + i_p;

						this->rates_off[i] = v_off <= 0 ? 0 :
								k_dens_fnc(sys_prms.ln_k0_off,
										   sys_prms.kA_off,
										   invlogS2) * v_off * this->rho_state[i];

						this->rates_on[i] =	k_dens_fnc(acetyl_state.ln_k0_on_table[i_k],
														  acetyl_state.kA_on_table[i_k],
														  invlogS2) *
											(acetyl_state.p_table[i_p] * n_sites_on);
						if(acetyl_state.NHhigh_arr[indpos_i] < N_chromatin){
							N_chr_onoff_ratio = acetyl_state.NHhigh_arr[indpos_i] / N_chromatin;
							i_p = acetyl_state.p_low_ind_arr[indpos_i];
							i_k = i_f * acetyl_state.N_p + i_p;
							this->rates_on[i] = this->rates_on[i] * N_chr_onoff_ratio +
									(1 - N_chr_onoff_ratio) *
									k_dens_fnc(acetyl_state.ln_k0_on_table[i_k],
											   acetyl_state.kA_on_table[i_k],
											   invlogS2) *
									(acetyl_state.p_table[i_p] * n_sites_on);
						}

//						++indpos_i;   // go to the next active site
					} else {
						this->rates_off[i] = k_dens_fnc(sys_prms.ln_k0_off, sys_prms.kA_off, sqr(this->invlogS_state[i])) * sys_prms.v_voxel * this->rho_state[i];
						this->rates_on[i] = 0;
					}
				}
			}

			if(acetyl_state.N_active_sites > indpos_i) if(acetyl_state.indpos_arr[indpos_i] == i) ++ indpos_i;
		}

		this->k_on_total = sum_array(this->rates_on, this->N_vox);
		this->k_off_total = sum_array(this->rates_off, this->N_vox);
	}

	void Proteins_state::step_Rs(double dt)
	{
		long indpos;
		for(long i = 0; i < this->N_droplets; ++i){
			indpos = this->Rs_indpos[i];
			this->Rs[indpos] += this->Rs_dot[i] * dt;

			// we assume droplets do not move => indpos stays the same

			// we assume droplets do not detach/attach to chromatin => is_on stays the same
		}

		this->update_R_powers();
	}

	void Proteins_state::generate_new_nuclei(double R, double dt)
	{
		double k, k_on, k_off;
		int is_on;
		double p_gen;
		for(long i = 0; i < this->N_vox; ++i){
			k_on = this->rates_on[i];
			k_off = this->rates_off[i];
			k = k_on + k_off;
			if(k > 0){
				p_gen = k * dt;
				if(p_gen > 1e-3 ? (gsl_rng_uniform(rng) > exp(-p_gen)) : (gsl_rng_uniform(rng) < p_gen)){
					if(k_on == 0){
						is_on = 0;
					} else if(k_off == 0){
						is_on = 1;
					} else {
						is_on = (gsl_rng_uniform(rng) < 1 / (1 + k_off / k_on));
					}

					this->add_droplet(R, i, is_on);
				}
			}
		}
	}

	long Proteins_state::merge_existing_nuclei(System_parameters & sys_prms, Acetylation_state & acetyl_state)
	{
		long n_merges = 0;
		if(this->N_droplets > 0){
			long indpos_i, indpos_j, indpos_new;
			double R_i, R_j;
			double rinv_ij;
			double pos_i[dim], pos_j[dim], pos_new[dim];
			for(long i = 0; i < this->N_droplets; ++i){
				indpos_i = this->Rs_indpos[i];
				R_i = this->Rs[indpos_i];
				indpos2pos(indpos_i, sys_prms.N_L, pos_i, sys_prms.l_voxel);

				for(long j = i + 1; j < this->N_droplets; ++j){
					indpos_j = this->Rs_indpos[j];
					R_j = this->Rs[indpos_j];
					indpos2pos(indpos_j, sys_prms.N_L, pos_j, sys_prms.l_voxel);

					rinv_ij = sys_prms.inv_pdist ? sys_prms.get_invdist_ordered(indpos_i, indpos_j) :
							  1/sqrt(sys_prms.PBC_mode == PBC_mode_no_PBC ? dist2_3d(pos_i, pos_j) :
									 dist2_3d_pbc(sys_prms.R, sys_prms.L, pos_i, pos_j));
					if((R_i + R_j) * rinv_ij > 1){
						if(this->droplet_on_inds[i] && this->droplet_on_inds[j]){   // this potentially can account for chromatin rearrangement
							indpos_new = this->R3[i] > this->R3[j] ? indpos_i : indpos_j;
						} else if((!this->droplet_on_inds[i]) && (!this->droplet_on_inds[j])){
							if(sys_prms.PBC_mode == PBC_mode_no_PBC){
								get_CoM(pos_i, this->R3[i], pos_j, this->R3[j], pos_new);
							} else {
								get_CoM_pbc(sys_prms.R, sys_prms.L, pos_i, this->R3[i], pos_j, this->R3[j], pos_new);
							}

							indpos_new = pos2indpos(pos_new, sys_prms.N_L, sys_prms.l_voxel);

							this->droplet_on_inds[j] = acetyl_state.indpos_is_acetylated(indpos_new);
						} else {
							indpos_new = this->droplet_on_inds[i] ? indpos_i : indpos_j;
						}

						this->Rs[indpos_i] = 0;

						this->Rs_indpos[j] = indpos_new;
						this->R3[j] = this->R3[i] + this->R3[j];
						this->Rs[indpos_j] = 0;
						this->Rs[indpos_new] = pow(this->R3[j], 1.0/3.0);
						this->R2[j] = sqr(this->Rs[indpos_new]);

						this->R3[i] = 0;   // so we can delete those later

						++n_merges;

//						printf("N_merged = %ld, indpos_i = %ld, onind[%ld] = %d, indpos_j = %ld, onind[%ld] = %d, indpos_new = %ld\n",
//							   n_merges, indpos_i, i, this->droplet_on_inds[i], indpos_j, j, this->droplet_on_inds[j], indpos_new);

						break;   // i-th droplet was merged into the j-th one, so quit checking the i-th droplet
					}
				}
			}

			long swap_ind = this->N_droplets;   // the droplet we will move (in the Rs[1...Ndroplets] space)
			do{
				-- swap_ind;
//			if(swap_ind < 0) break;
			}while(this->R3[swap_ind] == 0);   // find the 1st non-deleted nucleus

			long i = 0;   // search for R3==0
			while(i < swap_ind){
				if(this->R3[i] == 0){
					this->R3[i] = this->R3[swap_ind];
					this->R2[i] = this->R2[swap_ind];
					this->Rs_indpos[i] = this->Rs_indpos[swap_ind];
					this->droplet_on_inds[i] = this->droplet_on_inds[swap_ind];

					// this->R3[swap_ind] = 0;   // we don't need it since we will just realloc this position out

					do{
						-- swap_ind;
					}while((this->R3[swap_ind] == 0) && (i < swap_ind));
				}

				++i;
			}

//			this->realloc_Rs(swap_ind + 1);   // actually delete the right tail of the Rs
			this->N_droplets = swap_ind + 1;   // forget the right tail of the Rs
		}

		return n_merges;
	}

	void Proteins_state::update_rho_state(System_parameters & sys_prms)
	{
		double depletion;
		double inv_dist;
		double eps_V = 0.1 / sys_prms.V;
		double deplet_thr = (1 - eps_V * sys_prms.rho_coex) / (1 - sys_prms.rho_coex / this->rho_inf);
		double pos_vox[dim], pos_drop[dim];
		int depletion_warning_given = 0;
		long droplet_indpos;
		for(long i_vox = 0; i_vox < this->N_vox; ++i_vox){
			if(this->rho_state[i_vox] > 0){   // voxels already inside droplets already have rho=0
				depletion = 0;
				for(long i_drop = 0; i_drop < this->N_droplets; ++i_drop){
					droplet_indpos = this->Rs_indpos[i_drop];
					if(i_vox == droplet_indpos){   // the current voxel is the voxel of the droplet this->Rs_indpos[i_drop]
						depletion = -1;
						break;
					} else {
						if(sys_prms.inv_pdist){
							inv_dist = sys_prms.get_invdist(i_vox, droplet_indpos);
						} else {
							indpos2pos(droplet_indpos, sys_prms.N_L, pos_drop, sys_prms.l_voxel);
							indpos2pos(i_vox, sys_prms.N_L, pos_vox, sys_prms.l_voxel);
							inv_dist = 1.0 / sqrt(sys_prms.PBC_mode == PBC_mode_no_PBC ?
												  dist2_3d(pos_vox, pos_drop) :
												  dist2_3d_pbc(sys_prms.R, sys_prms.L, pos_vox, pos_drop));
						}
					}

					if(inv_dist * this->Rs[droplet_indpos] > 1){   // the current voxel is inside the droplet this->Rs_indpos[i_drop]
						depletion = -1;
						break;
					} else {
						depletion += this->dR[i_drop] * inv_dist;
					}
				}

				if(depletion >= 0) {
//					this->rho_state[i_vox] = depletion < deplet_thr ? this->rho_inf * (1 - depletion) : eps_V;
					this->rho_state[i_vox] = depletion < deplet_thr ? (sys_prms.rho_coex + (this->rho_inf - sys_prms.rho_coex) * (1 - depletion)) : eps_V;
					if(verbose_default > 0){
						if((depletion >= deplet_thr) && (!depletion_warning_given)){
							printf("WARNING: depletion for indpos_vox = %ld is %lf >= %lf\n", i_vox, depletion, deplet_thr);
							depletion_warning_given = 1;
						}
					}
				} else {
					this->rho_state[i_vox] = 0;   // mark voxels inside droplets
				}
			}
		}
	}

	System_parameters::System_parameters(const long* N_L_new, int PBC_mode_new, double rho_coex_new, double rho_l_new,
										 double R_S0_new, double S0_new, double D_new, double l_voxel_new, double R_0_new,
										 double ln_k0_off_new, double kA_off_new, int to_precomp_dist, int R_crit_mode_new,
										 int accuracy_mode_new, int verbose_new):
			rho_coex(rho_coex_new), rho_l(rho_l_new), R_S0(R_S0_new * log(S0_new)), S0(S0_new), D(D_new), l_voxel(l_voxel_new), R_0(R_0_new),
			v_voxel(powi(l_voxel_new, dim)), ln_k0_off(ln_k0_off_new), kA_off(kA_off_new), PBC_mode(PBC_mode_new),
			accuracy_mode(accuracy_mode_new), R_crit_mode(R_crit_mode_new), verbose(verbose_new)
	{
		this->change_system_size(N_L_new, to_precomp_dist);

//		if(this->R_0 < 0){
//			throw std::runtime_error("ERROR: smart R_0 (input R_0 < 0) not supported yet\n");
//		}
	}

	System_parameters::System_parameters(py_c_arr(long) & N_L_new, int PBC_mode_new, double rho_coex_new, double rho_l_new,
										 double R_S0_new, double S0_new, double D_new, double l_voxel_new, double R_0_new,
										 double ln_k0_off_new, double kA_off_new, int to_precomp_dist, int R_crit_mode_new,
										 int accuracy_mode_new, int verbose_new):
			System_parameters(extract_arr_from_pybind11(N_L_new, dim, 1),
							  PBC_mode_new, rho_coex_new, rho_l_new, R_S0_new, S0_new, D_new, l_voxel_new,
							  R_0_new, ln_k0_off_new, kA_off_new, to_precomp_dist, R_crit_mode_new,
							  accuracy_mode_new, verbose_new)
	{}

	void System_parameters::change_system_size(const long* N_L_new, int to_precomp_dist)
	{
		memcpy(this->N_L, N_L_new, sizeof(long) * dim);
		this->N_vox = prod_array(this->N_L, dim);
		this->V = this->N_vox * this->v_voxel;
		this->Rmax = pow(3 * this->V / (4 * M_PI), 1.0 / 3);
		FOR_DIM{
			this->L[_i_dim] = this->N_L[_i_dim] * this->l_voxel;
			this->R[_i_dim] = this->L[_i_dim] / 2;
		}

		if(to_precomp_dist){
			c_malloc(& this->inv_pdist, this->N_vox * (this->N_vox - 1) / 2);
			double *voxels_pos = c_malloc<double>(dim * this->N_vox);

			for(long i = 0; i < this->N_vox; ++i){
				indpos2pos(i, this->N_L, &(voxels_pos[dim * i]), this->l_voxel);
			}

			if(this->PBC_mode == PBC_mode_no_PBC){
				for(long i = 0; i < this->N_vox; ++i){
					for(long j = i + 1; j < this->N_vox; ++j){
						this->set_invdist_ordered(i, j, 1 / sqrt(dist2_3d(&(voxels_pos[i * dim]), &(voxels_pos[j * dim]))));
					}
				}
			} else {
				for(long i = 0; i < this->N_vox; ++i){
					for(long j = i + 1; j < this->N_vox; ++j){
						this->set_invdist_ordered(i, j, 1 / sqrt(dist2_3d_pbc(this->R, this->L, &(voxels_pos[i * dim]), &(voxels_pos[j * dim]))));
					}
				}
			}

			c_free(& voxels_pos);
		}
	}

	void System_parameters::print_params()
	{
		printf("N_vox = %ld, N_L = (%ld, %ld, %ld), rho_coex = %lf, rho_l = %lf, R_S0 = %lf, S0 = %lf, D = %lf, l_voxel = %lf, R_0 = %lf, ln_k0_off = %lf, kA_off = %lf\n",
			   this->N_vox, this->N_L[0], this->N_L[1], this->N_L[2], this->rho_coex, this->rho_l, this->R_S0, this->S0, this->D, this->l_voxel, this->R_0, this->ln_k0_off, this->kA_off);
	}

	Acetylation_state::Acetylation_state(long N_active_sites_new, double D_Hbig_new, double D_H_new, long N_vox_new,
										 long *indpos_arr_new,
										 long *NHhigh_arr_new, long *p_high_ind_arr_new, long *p_low_ind_arr_new,
										 long *f_inv_ind_arr_new, double *_ln_k0_on, double *_kA_on, double *_p_arr,
										 double *_f_inv_arr, long N_p_new, long N_f_new)
		: N_active_sites(N_active_sites_new), N_vox(N_vox_new), D_Hbig(D_Hbig_new), D_H(D_H_new),
		  indpos_arr(indpos_arr_new), NHhigh_arr(NHhigh_arr_new),
		  p_high_ind_arr(p_high_ind_arr_new), p_low_ind_arr(p_low_ind_arr_new), f_inv_ind_arr(f_inv_ind_arr_new),
		  ln_k0_on_table(_ln_k0_on), kA_on_table(_kA_on), p_table(_p_arr), f_inv_table(_f_inv_arr), N_p(N_p_new), N_f(N_f_new),
		  alloced_arrs{0}
	{}

	Acetylation_state::Acetylation_state(const long *N_L, long N_active_sites_new, double D_Hbig_new, double D_H_new,
										 long N_vox_new,
										 int p_gen_mode, long p_high_ind, long NH_high, long p_low_ind, long f_inv_ind,
										 double *_ln_k0_on, double *_kA_on, double *_p_arr,
										 double *_f_inv_arr, long N_p_new, long N_f_new)
			: N_active_sites(N_active_sites_new), N_vox(N_vox_new), D_Hbig(D_Hbig_new), D_H(D_H_new),
			  ln_k0_on_table(_ln_k0_on), kA_on_table(_kA_on), p_table(_p_arr), f_inv_table(_f_inv_arr), N_p(N_p_new), N_f(N_f_new)
	{
		this->generate(N_L, p_gen_mode, p_high_ind, NH_high, p_low_ind, f_inv_ind, 1);
	}

	Acetylation_state::Acetylation_state(long N_active_sites_new, double D_Hbig_new, double D_H_new, long N_vox_new,
										 py_c_arr(long) & indpos_arr_new,
										 py_c_arr(long) & NHhigh_arr_new, py_c_arr(long) & p_high_ind_arr_new,
										 py_c_arr(long) & p_low_ind_arr_new, py_c_arr(long) & f_inv_ind_arr_new,
										 py_c_arr(double) &_ln_k0_on, py_c_arr(double) &_kA_on,
										 py_c_arr(double) &_p_arr, py_c_arr(double) &_f_inv_arr):
			Acetylation_state(N_active_sites_new, D_Hbig_new, D_H_new, N_vox_new,
							  extract_arr_from_pybind11(indpos_arr_new, N_active_sites_new, 1),
							  extract_arr_from_pybind11(NHhigh_arr_new, N_active_sites_new, 1),
							  extract_arr_from_pybind11(p_high_ind_arr_new, N_active_sites_new, 1),
							  extract_arr_from_pybind11(p_low_ind_arr_new, N_active_sites_new, 1),
							  extract_arr_from_pybind11(f_inv_ind_arr_new, N_active_sites_new, 1),
							  extract_arr_from_pybind11(_ln_k0_on, _p_arr.shape()[0]  * _f_inv_arr.shape()[0], 1),
							  extract_arr_from_pybind11(_kA_on, _p_arr.shape()[0]  * _f_inv_arr.shape()[0], 1),
							  extract_arr_from_pybind11(_p_arr, _p_arr.shape()[0], 1),
							  extract_arr_from_pybind11(_f_inv_arr, _f_inv_arr.shape()[0], 1),
							  _p_arr.shape()[0], _f_inv_arr.shape()[0])
	{}

	Acetylation_state::~Acetylation_state()
	{
		if(this->alloced_arrs){
			this->free_arrs();
		}
	}

	void Acetylation_state::free_arrs()
	{
		c_free(& this->indpos_arr);
		c_free(& this->NHhigh_arr);
		c_free(& this->p_high_ind_arr);
		c_free(& this->p_low_ind_arr);
		c_free(& this->f_inv_ind_arr);
	}

	void Acetylation_state::alloc_arrs()
	{
		c_malloc(& this->indpos_arr, this->N_active_sites);
		c_malloc(& this->NHhigh_arr, this->N_active_sites);
		c_malloc(& this->p_high_ind_arr, this->N_active_sites);
		c_malloc(& this->p_low_ind_arr, this->N_active_sites);
		c_malloc(& this->f_inv_ind_arr, this->N_active_sites);
		this->alloced_arrs = 1;
	}

	void Acetylation_state::sort_by_indpos()
	{
		long *sort_inds;
//		c_malloc(& sort_inds, this->N_active_sites);

		argsort(this->indpos_arr, & sort_inds, this->N_active_sites, 1);

		reorder(this->indpos_arr, sort_inds, this->N_active_sites);
		reorder(this->NHhigh_arr, sort_inds, this->N_active_sites);
		reorder(this->p_high_ind_arr, sort_inds, this->N_active_sites);
		reorder(this->p_low_ind_arr, sort_inds, this->N_active_sites);
		reorder(this->f_inv_ind_arr, sort_inds, this->N_active_sites);

		c_free(& sort_inds);
	}

	void Acetylation_state::print_params() const
	{
		printf("N_active_sites = %ld, N_p = %ld, N_f = %ld, N_vox = %ld\n",
			   this->N_active_sites, this->N_p, this->N_f, this->N_vox);
	}

	int Acetylation_state::indpos_is_acetylated(long indpos) const
	{
		for(long i = 0; i < this->N_active_sites; ++i){
			if(this->indpos_arr[i] == indpos){
				return 1;
			}
		}

		return 0;
	}

	void Acetylation_state::generate(const long *N_L, int p_gen_mode, long p_high_ind, long NH_high, long p_low_ind,
									 long f_inv_ind, int to_alloc)
	{
		if(to_alloc) this->alloc_arrs();

		//   N_vox_active = int(N_vox * active_vox_dens + 0.5)

		for(long i = 0; i < this->N_active_sites; ++i){
			this->p_high_ind_arr[i] = p_high_ind;
			this->p_low_ind_arr[i] = p_low_ind;
			this->f_inv_ind_arr[i] = f_inv_ind;
			this->NHhigh_arr[i] = NH_high;
		}

		long *indpos_buff;
		switch (p_gen_mode) {
			case p_gen_mode_shuffle:
				c_malloc(& indpos_buff, this->N_vox);
				for(long i = 0; i < this->N_vox; ++i){
					indpos_buff[i] = i;
				}
				gsl_ran_choose (rng, this->indpos_arr, this->N_active_sites, indpos_buff, this->N_vox, sizeof (long));
				c_free(& indpos_buff);

				break;

			case p_gen_mode_1site:
				if(this->N_active_sites != 1){
					printf("Inconsistent p_gen_mode = %d and N_active_sites = %ld (must be 1 for this mode)", p_gen_mode, this->N_active_sites);
					throw std::runtime_error("Inconsistent p_gen_mode and N_active_sites");
				}

				this->indpos_arr[0] = pos2indpos(N_L[0] / 2, N_L[1] / 2, N_L[2] / 2, N_L);

				break;

			case p_gen_mode_2neibrs:
				if(this->N_active_sites != 2){
					printf("Inconsistent p_gen_mode = %d and N_active_sites = %ld (must be 2 for this mode)", p_gen_mode, this->N_active_sites);
					throw std::runtime_error("Inconsistent p_gen_mode and N_active_sites");
				}

				this->indpos_arr[0] = pos2indpos(N_L[0] / 2, N_L[1] / 2, N_L[2] / 2, N_L);
				this->indpos_arr[1] = this->indpos_arr[0] - 1;

				break;

			default:
				printf("p_gen_mode = %d\n", p_gen_mode);
				throw std::runtime_error("This p_gen_mode is not supported");
		}
	}

	void Run_parameters::print_params() const
	{
		printf("Nt_max = %ld, equil_mode = %d, dt = %lf, thermo_stride = %ld, Rs_stride = %ld, rho_stride = %ld\n",
			   this->Nt_max, this->equil_mode, this->dt, this->thermo_stride, this->Rs_stride, this->rho_stride);
	}


//	bool is_potential_swap_position(int *state, int L, int ix, int iy)
//	{
//		int s_group[2 * dim + 1];
//		get_spin_with_neibs(state, L, ix, iy, s_group);
//		for(int j = 1; j <= 2 * dim; ++j){
//			if(s_group[0] != s_group[j]){
//				return true;
//			}
//		}
//
//		return false;
//	}
//
//	void update_potential_position(int *state, int L, int pos, std::set< int > *positions){
//		auto pos_pos = positions->find(pos);
//		if(pos_pos == positions->end()){
//			if(is_potential_swap_position(state, L, pos / L, pos % L)){
//				positions->insert(pos);
//			}
//		} else {
//			if(!is_potential_swap_position(state, L, pos / L, pos % L)){
//				positions->erase(pos_pos);
//			}
//		}
//
//	}
//
//	void update_neib_potpos(int *state, int L, int ix, int iy, std::set< int > *positions)
//	{
//		int L2 = L*L;
//		int pos = ix * L + iy;
//		update_potential_position(state, L, md(ix + 1, L) * L + iy, positions);
//		update_potential_position(state, L, md(ix - 1, L) * L + iy, positions);
//		update_potential_position(state, L, ix * L + md(iy + 1, L), positions);
//		update_potential_position(state, L, ix * L + md(iy - 1, L), positions);
//	}
//
//	void find_potential_swaps(int *state, int L, std::set< int > *positions)
//	/*
//	 * This algorithm is not optimal
//	 * A more optimal way would be to cluster the state with a criterion "g_0 != g_neig"
//	 */
//	{
//		int L2 = L*L;
//		long i, j;
//		int s_group[2 * dim + 1];
//		for(i = 0; i < L2; ++i){
//			if(is_potential_swap_position(state, L, i / L, i % L)){
//				positions->insert(i);
//			}
//		}
//	}

	double Proteins_state::comp_Va() const
	{
		double R3_sum = 0;
		for(long i = 0; i < this->N_droplets; ++i){
			R3_sum += this->R3[i];
		}

		return R3_sum * (4.0 * M_PI / 3.0);
	}

	double Proteins_state::comp_Sj(System_parameters & sys_prms, long j) const
	{
		long indpos = this->Rs_indpos[j];
		double droplet_pos[dim];
		indpos2pos(indpos, sys_prms.N_L, droplet_pos, sys_prms.l_voxel);

		double R_max;
		if(sys_prms.PBC_mode == PBC_mode_no_PBC){
			R_max = std::max(this->Rs[indpos],
							 std::min({droplet_pos[0], sys_prms.N_L[0] - droplet_pos[0],
							 			 droplet_pos[1], sys_prms.N_L[1] - droplet_pos[1],
							  			 droplet_pos[2], sys_prms.N_L[2] - droplet_pos[2]})
						   );
		} else {
			R_max = sys_prms.Rmax;
		}

		double Sj = (sqr(R_max) - sqr(this->Rs[indpos])) * (2 * M_PI);

		if(sys_prms.accuracy_mode & accuracy_mode_excl_nucls){
			double other_droplets_excl = 0;
			if(sys_prms.inv_pdist){
				for(long i = 0; i < j; ++i){
					other_droplets_excl += this->R3[i] * sys_prms.get_invdist(this->Rs_indpos[i], indpos);
				}
				for(long i = j + 1; i < this->N_droplets; ++i){
					other_droplets_excl += this->R3[i] * sys_prms.get_invdist(this->Rs_indpos[i], indpos);
				}
			} else {
				throw std::runtime_error("No precomputed invdist - not supported");
			}
			Sj -= (other_droplets_excl * (4 * M_PI / 3));
		}

		if(sys_prms.accuracy_mode & accuracy_mode_integr_outside){
			double rinv_integral = 0;
			double Rmax_inv = 1 / R_max;
			if(sys_prms.PBC_mode == PBC_mode_no_PBC){
				double invdist;
				for(long i = 0; i < indpos; ++i){
					invdist = sys_prms.get_invdist_ordered(i, indpos);
					if(invdist < Rmax_inv){
						rinv_integral += invdist;
					}
				}
				for(long i = indpos + 1; i < sys_prms.N_vox; ++i){
					invdist = sys_prms.get_invdist_ordered(indpos, i);
					if(invdist < Rmax_inv){
						rinv_integral += invdist;
					}
				}
				rinv_integral *= sys_prms.v_voxel;
				Sj += rinv_integral;
			}
		}

		return Sj > 0 ? Sj : 0;
	}

	double Proteins_state::comp_Vb(System_parameters & sys_prms) const
	{
		double Vb_local = 0;
		for(long i = 0; i < this->N_droplets; ++i){
			Vb_local += this->dR[i] * this->Sj[i];  // this->comp_Sj(sys_prms, i);
		}

		return Vb_local;
	}

	void Proteins_state::comp_Vb_s(voxels::System_parameters &sys_prms, double *Vb0_local, double *Vc_local) const
	{
		*Vb0_local = 0;
		*Vc_local = 0;
		long indpos;
		for(long i = 0; i < this->N_droplets; ++i){
			indpos = this->Rs_indpos[i];
			*Vb0_local += this->Rs[indpos] * this->Sj[i];
			*Vc_local += this->Sj[i];
		}

		*Vc_local *= sys_prms.R_S0;
	}

	void Proteins_state::update_Va()
	{
		this->Va = this->comp_Va();
	}

	void Proteins_state::update_Vb(System_parameters & sys_prms)
	{
		this->Vb = this->comp_Vb(sys_prms);
	}

	void Proteins_state::update_Vbs(System_parameters & sys_prms)
	{
		this->comp_Vb_s(sys_prms, &(this->Vb0), &(this->Vc));
	}

	void Proteins_state::update_Vdots(voxels::System_parameters & sys_prms)
	{
		for(long i = 0; i < this->N_droplets; ++i){
			this->Sj_dot[i] = this->Rs[this->Rs_indpos[i]] * this->Rs_dot[i];

			if(sys_prms.accuracy_mode & accuracy_mode_excl_nucls){
				for(long j = 0; j < i; ++j){
					this->Sj_dot[i] += this->R2[j] * this->Rs_dot[j] * sys_prms.get_invdist(this->Rs_indpos[i], this->Rs_indpos[j]);
				}
				for(long j = i + 1; j < this->N_droplets; ++j){
					this->Sj_dot[i] += this->R2[j] * this->Rs_dot[j] * sys_prms.get_invdist(this->Rs_indpos[i], this->Rs_indpos[j]);
				}
			}

			this->Sj_dot[i] *= (- 4 * M_PI);
		}

		this->Va_dot = 0;
		for(long i = 0; i < this->N_droplets; ++i){
			this->Va_dot += this->R2[i] * this->Rs_dot[i];
		}
		this->Va_dot *= (4 * M_PI);

		if(sys_prms.R_crit_mode == R_crit_mode_const){
			this->Vb_dot = 0;

			for(long i = 0; i < this->N_droplets; ++i){
				this->Vb_dot += (this->Sj_dot[i] * this->dR[i] + this->Sj[i] * this->Rs_dot[i]);
			}

			this->rho_inf_dot = (((this->Vb_dot * sys_prms.rho_coex - this->Va_dot * sys_prms.rho_l) * (sys_prms.V + this->Vb - this->Va)) -
								(this->Vb_dot - this->Va_dot) * (this->Vb * sys_prms.rho_coex - this->Va * sys_prms.rho_l)) /
								sqr(sys_prms.V + this->Vb - this->Va);
		} else if(sys_prms.R_crit_mode == R_crit_mode_CNT){
			this->Vc_dot = sys_prms.R_S0 * sum_array(this->Sj_dot, this->N_droplets);

			this->Vb0_dot = 0;
			for(long i = 0; i < this->N_droplets; ++i){
				this->Vb0_dot += (this->Rs_dot[i] * this->Sj[i] + this->Sj_dot[i] * this->Rs[this->Rs_indpos[i]]);
			}

			double b = (sys_prms.V - this->Va + this->Vb0) / this->Vc;
			double c = (sys_prms.protein_total_amount / sys_prms.rho_coex - sys_prms.V - this->Va * (sys_prms.rho_l / sys_prms.rho_coex - 1)) / this->Vc;

			double b_dot = ((this->Vb0_dot - this->Va_dot) * this->Vc - this->Vc_dot * (sys_prms.V - this->Va + this->Vb0)) / (this->Vc * this->Vc);
			double c_dot = -(this->Vc * this->Va_dot * (sys_prms.rho_l / sys_prms.rho_coex - 1) +
							this->Vc_dot * (sys_prms.protein_total_amount / sys_prms.rho_coex - sys_prms.V - this->Va*(sys_prms.rho_l / sys_prms.rho_coex - 1))) /
							(this->Vc * this->Vc);

			double sgm_inf = this->rho_inf / sys_prms.rho_coex - 1;
			double invLnS = 1/log(1 + sgm_inf);

			this->rho_inf_dot = sys_prms.rho_coex * ((c_dot * (b - invLnS) - b_dot * c) / (sqr(b - invLnS) + c/(1 + sgm_inf) * invLnS * invLnS));
		}
	}

	double Proteins_state::get_good_dt(System_parameters & sys_prms)
	{
		double *relevant_timescales;
		c_malloc(& relevant_timescales, this->N_droplets + 2);

		relevant_timescales[0] = 1 / (this->k_on_total + this->k_off_total);
		relevant_timescales[1] = abs((this->rho_inf - sys_prms.rho_coex) / this->rho_inf_dot);
		for(long i = 0; i < this->N_droplets; ++i){
			relevant_timescales[i + 2] = this->Rs[this->Rs_indpos[i]] / abs(this->Rs_dot[i]);
		}

		double min_timescale = min_array(relevant_timescales, this->N_droplets + 2);

		if((min_timescale <= 0) || (min_timescale > 1e15)){
			printf("k_on_tot = %e, k_off_tot = %e, rho_inf = %lf, rho_inf_dot = %e, rho_inf / rho_coex - 1 = %e, Vb = %e, V+Vb-Va = %e, Vb_dot = %e, Va_dot = %e\n",
				   this->k_on_total, this->k_off_total, this->rho_inf, this->rho_inf_dot, this->rho_inf / sys_prms.rho_coex - 1, this->Vb, sys_prms.V + this->Vb - this->Va, this->Vb_dot, this->Va_dot);

			printf("%10s, %10s, %10s, %10s, %10s, %10s\n", "j", "Rs", "Rs_dot", "dR", "Sj", "Sj_dot");
			for(long i = 0; i < this->N_droplets; ++i){
				printf("%10ld %10e %10e, %10e %10e %10e\n", i, this->Rs[this->Rs_indpos[i]], this->Rs_dot[i], this->dR[i], this->Sj[i], this->Sj_dot[i]);
			}

			printf("dt_scale = %e; ", min_timescale);
			for(long i = 0; i < this->N_droplets + 2; ++i){
				printf("%e ", relevant_timescales[i]);
			}
			printf("\n");
			printf("\n");
			STP
		}

		c_free(& relevant_timescales);

		return min_timescale;
	}

	void Proteins_state::thermo_log_dump(long Nt, long N_merges, double time, Thermo_log & thermo_log, System_parameters & sys_prms)
	{
		thermo_log.add_record(this->rho_inf / sys_prms.rho_coex,
							  this->N_droplets,
							  sum_array(this->droplet_on_inds, this->N_droplets),
							  N_merges,
							  Nt,
							  time,
							  this->k_off_total,
							  this->k_on_total,
							  this->Va,
							  sys_prms.R_crit_mode == R_crit_mode_CNT ? this->Vb0 : this->Vb,
							  sys_prms.R_crit_mode == R_crit_mode_CNT ? this->Vc : 0,
							  sys_prms.verbose);
	}

	void Proteins_state::Rs_log_dump(double time, voxels::Rs_log &rs_log, voxels::System_parameters &sys_prms)
	{
		rs_log.add_record(time,
						  this->N_droplets,
						  this->Rs,
						  this->Rs_indpos,
						  this->droplet_on_inds,
						  sys_prms.verbose);
	}

	void Proteins_state::Rho_log_dump(double time, voxels::Rho_log &rho_log, voxels::System_parameters &sys_prms)
	{
		rho_log.add_record(time,
						   this->rho_state,
						   sys_prms.verbose);
	}

	int Proteins_state::run_state(long *Nt,
				  Acetylation_state & acetylation_state,
				  System_parameters & sys_prms,
				  Run_parameters & run_prms,
				  Thermo_log & thermo_log,
				  Rs_log & rs_log,
				  Rho_log & rho_log)
	{
		long N_merges_by_now = 0;
		double time = 0;
		double dt_local = run_prms.dt;
		while(true){
			switch (sys_prms.R_crit_mode) {
				case R_crit_mode_const:
					this->update_dR(sys_prms);
					this->update_rho_inf(sys_prms);
					break;
				case R_crit_mode_CNT:
					this->update_rho_inf(sys_prms);
					this->update_dR(sys_prms);
					break;
			}
			this->update_rho_state(sys_prms);
			this->update_logS(sys_prms.rho_coex, 1.0 / sys_prms.V);
			this->update_Rdot(sys_prms);
			this->update_rates(sys_prms, acetylation_state);
			if(this->to_comp_dots){
				this->update_Vdots(sys_prms);
				dt_local = this->get_good_dt(sys_prms) * (-run_prms.dt);
			}

			/* previous step is finished */

			// ------------------ save timeevol ----------------
			if(thermo_log.stride > 0){
				if((*Nt) % thermo_log.stride == 0){
					this->thermo_log_dump(*Nt, N_merges_by_now, time, thermo_log, sys_prms);
				}
			}

			if(rs_log.stride > 0){
				if((*Nt) % rs_log.stride == 0){
					this->Rs_log_dump(time, rs_log, sys_prms);
				}
			}

			if(rho_log.stride > 0){
				if((*Nt) % rho_log.stride == 0){
					Rho_log_dump(time, rho_log, sys_prms);
				}
			}

			/* previous step is logged */

			// ---------------- check BF exit ------------------
			int quit_status = 0;
			if(run_prms.Nt_max > 0){
				if(*Nt >= run_prms.Nt_max){
					if(sys_prms.verbose){
						if(sys_prms.verbose >= 2) {
							printf("Reached desired Nt >= Nt_max (= %ld)                              \n", run_prms.Nt_max);
						} else {
							printf("\n");
						}
					}

					quit_status += quit_status_Ntmax;
				}
			}

			if(run_prms.S_thr > 0){
				if(this->rho_inf / sys_prms.rho_coex < run_prms.S_thr){
					if(sys_prms.verbose){
						printf("Reached desired S < S_thr (= %lf)                                        \n", run_prms.S_thr);
					} else {
						printf("\n");
					}

					quit_status += quit_status_Sthr;
				}
			}

			if(run_prms.k_tot_thr > 0){
				if(this->k_on_total + this->k_off_total < run_prms.k_tot_thr){
					if(sys_prms.verbose){
						printf("Reached desired k_total < k_tot_thr (= %e)                                 \n", run_prms.k_tot_thr);
					} else {
						printf("\n");
					}

					quit_status += quit_status_Ktot;
				}
			}

			if(quit_status > 0){
				if(thermo_log.stride > 0){
					if((*Nt) % thermo_log.stride != 0){   // we do not want to save the final state if we have already saved the same state during the current run
						this->thermo_log_dump(*Nt, N_merges_by_now, time, thermo_log, sys_prms);
					}
				}

				if(rs_log.stride > 0){
					if((*Nt) % rs_log.stride != 0){
						this->Rs_log_dump(time, rs_log, sys_prms);
					}
				}

				if(rho_log.stride > 0){
					if((*Nt) % rho_log.stride != 0){
						Rho_log_dump(time, rho_log, sys_prms);
					}
				}

				return quit_status;
			}

			/* we have quit if we needed to, otherwise - going to the next step */

			// ----------- modify stats -----------

			this->step_Rs(dt_local);
			if(this->rho_inf > sys_prms.rho_coex){
				this->generate_new_nuclei(sys_prms.R_0 > 0 ? sys_prms.R_0 :
										  (-sys_prms.R_0 *
										  	(sys_prms.R_crit_mode == R_crit_mode_const ?
											  sys_prms.R_S0 / log(sys_prms.S0) :
											  sys_prms.R_S0 / log(this->rho_inf / sys_prms.rho_coex))),
										  dt_local);
			}
			N_merges_by_now += this->merge_existing_nuclei(sys_prms, acetylation_state);

			// this->attach_off_nucl_to_active_sites(..); // move off-nucl to on-sites within nuclei

			++(*Nt);
			time += dt_local;

			if(sys_prms.verbose){
				if(!(*Nt % (1000))){
					printf("BF run: Nt = %ld", *Nt);
					if(this->to_comp_dots){
						printf("; dt = %lf", dt_local);
					}
					if(run_prms.Nt_max > 0){
						printf("; %lf %%", (double)(*Nt) / (run_prms.Nt_max) * 100);
					}
					if(run_prms.S_thr > 0){
						printf("; S = %lf, S_thr = %lf", this->rho_inf / sys_prms.rho_coex, run_prms.S_thr);
					}
					if(run_prms.k_tot_thr > 0){
						printf("; k_on_tot = %e, k_off_tot = %e, k_tot_thr = %e", this->k_on_total, this->k_off_total, run_prms.k_tot_thr);
					}
					printf("                         \r");
					fflush(stdout);
				}
			}
		}
	}

	int run_bruteforce_C(long *Nt,
						 Proteins_state & prot_init_state,
						 Acetylation_state & acetylation_state,
						 System_parameters & sys_prms,
						 Run_parameters & run_prms,
						 Thermo_log & thermo_log,
						 Rs_log & rs_log,
						 Rho_log & rho_log)
	{
		if(sys_prms.verbose){
			prot_init_state.print_params();
			acetylation_state.print_params();
			sys_prms.print_params();
			run_prms.print_params();
			thermo_log.print_params();
			rs_log.print_params();
			rho_log.print_params();
		}

		Proteins_state state_under_process(& prot_init_state);

		while(true){
//			state_under_process.copy_from(rho_log.get_ith_state(restart_state_ID));

			sys_prms.protein_total_amount = state_under_process.protein_total_amount(sys_prms.l_voxel);
			state_under_process.run_state(Nt, acetylation_state, sys_prms, run_prms, thermo_log, rs_log, rho_log);

			break;
		}

		state_under_process.clear();

		return 0;
	}

	int get_equilibrated_state(Proteins_state & prot_init_state,
							   Acetylation_state & acetylation_state,
							   System_parameters & sys_prms,
							   Run_parameters & run_prms,
							   Proteins_state * prot_equild_state)
	{
		switch (run_prms.equil_mode) {
			case equil_mode_no_eq:
				prot_equild_state->clear();
				prot_equild_state->copy_from(& prot_init_state, 1);
		}

		return 0;
	}

//
//	void get_spin_with_neibs(const int *state, int L, int ix, int iy, int *s_group)
//	{
//		s_group[0] = state[ix * L + iy];
//		s_group[1] = state[md(ix + 1, L)*L + iy];
//		s_group[2] = state[ix*L + md(iy + 1, L)];
//		s_group[3] = state[md(ix - 1, L)*L + iy];
//		s_group[4] = state[ix*L + md(iy - 1, L)];
//	}

	int init_rand_C(long my_seed)
	/**
	 * Sets up a new GSL randomg denerator seed
	 * @param my_seed - the new seed for GSL
	 * @return  - the Error code
	 */
	{
		// initialize random generator
		gsl_rng_env_setup();
		const gsl_rng_type* T = gsl_rng_default;
		rng = gsl_rng_alloc(T);
		gsl_rng_set(rng, my_seed);
//		srand(my_seed);

//		gen_mt19937 = new std::mt19937( std::random_device{}() );

		seed = my_seed;
		return 0;
	}

//    int print_S(const int *s, int L, char prefix)
//    {
//        long i, j;
//
//        if(prefix > 0){
//            printf("%c\n", prefix);
//        }
//
//        for(i = 0; i < L; ++i){
//            for(j = 0; j < L; ++j){
//                printf("%2d", s[i*L + j]);
//            }
//            printf("\n");
//        }
//
//        return 0;
//    }

//	int state_is_valid(const int *s, int L, int k, char prefix)
//	{
//		for(long i = 0; i < L*L; ++i) if(abs(s[i]) != 1) {
//				printf("%d\n", k);
//				print_S(s, L, prefix);
//				return 0;
//			}
//		return 1;
//	}

//	int is_neib_x(int L, int i, int j)
//	{
//		int d = abs(i - j);
//		return std::min(d, L-d) <= 1;
//	}
//
//	int is_neib(int L, int ix1, int iy1, int ix2, int iy2, int mode=1)
//	{
//		if(mode == 1){
//			/*
//			 * visual neibrs:
//			 * ***
//			 * *o*
//			 * ***
//			 */
//			return is_neib_x(L, ix1, ix2) && is_neib_x(L, iy1, iy2);
//		}
//	}

}