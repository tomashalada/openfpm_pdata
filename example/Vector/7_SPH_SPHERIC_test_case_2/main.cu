/*! \page Vector_7_sph_dlb_gpu_more_opt Vector 7 SPH Dam break simulation with Dynamic load balacing on Multi-GPU (more optimized version)
 *
 *
 */

#ifdef __NVCC__

#define PRINT_STACKTRACE
#define STOP_ON_ERROR
#define OPENMPI
#define SCAN_WITH_CUB
#define SORT_WITH_CUB
//#define SE_CLASS1

//#define USE_LOW_REGISTER_ITERATOR

#include "Vector/vector_dist.hpp"
#include <math.h>
#include "Draw/DrawParticles.hpp"

#include "defs.h"
typedef float real_number;

// A constant to indicate boundary particles
#define BOUNDARY 0

// A constant to indicate fluid particles
#define FLUID 1

// initial spacing between particles dp in the formulas
const real_number dp = 0.02;
// Maximum height of the fluid water
// is going to be calculated and filled later on
real_number h_swl = 0.0;

// c_s in the formulas (constant used to calculate the sound speed)
const real_number coeff_sound = 20.0;

// gamma in the formulas
const real_number gamma_ = 7.0;

// Hcoef = sqrt(3) or Hcoef = 2
const real_number Hcoef = 2.0;
// Hcoef * dp support of the kernel
const real_number H = Hcoef * dp;

// Eta in the formulas
const real_number Eta2 = 0.01 * H*H;

const real_number FourH2 = 4.0 * H*H;

// alpha in the formula
const real_number visco = 0.1;

// Reference densitu 1000Kg/m^3
const real_number rho_zero = 1000.0;

// cbar in the formula (calculated later)
real_number cbar = 0.0;

// Mass of the fluid particles
const real_number MassFluid = rho_zero * dp * dp * dp;

// Mass of the boundary particles
const real_number MassBound = rho_zero * dp * dp * dp;

// End simulation time
//#ifdef TEST_RUN
//const real_number t_end = 0.001;
//#else
const real_number t_end = 1.0;
//#endif

// Gravity acceleration
const real_number gravity = 9.81;

// Filled later require h_swl, it is b in the formulas
real_number B = 0.0;

// Constant used to define time integration
const real_number CFLnumber = 0.2;

// Minimum T
const real_number DtMin = 1.0e-9;

// Minimum Rho allowed
const real_number RhoMin = 700.0;

// Maximum Rho allowed
const real_number RhoMax = 1300.0;

// Filled in initialization
real_number max_fluid_height = 0.0;

// Properties

// FLUID or BOUNDARY
const size_t type = 0;

// Density
const int rho = 1;

// Density at step n-1
const int rho_prev = 2;

// Pressure
const int Pressure = 3;

// Delta rho calculated in the force calculation
const int drho = 4;

// calculated force
const int force = 5;

// velocity
const int velocity = 6;

// velocity at previous step
const int velocity_prev = 7;

const int red = 8;

const int red2 = 9;

// Type of the vector containing particles
typedef vector_dist_gpu<3,real_number,aggregate<unsigned int,real_number,  real_number,    real_number,     real_number,     real_number[3], real_number[3], real_number[3], real_number, real_number>> particles;
//                                              |          |             |               |                |                |               |               |               |            |
//                                              |          |             |               |                |                |               |               |               |            |
//                                             type      density       density        Pressure          delta            force          velocity        velocity        reduction     another
//                                                                     at n-1                           density                                         at n - 1        buffer        reduction buffer


struct ModelCustom
{
	template<typename Decomposition, typename vector> inline void addComputation(Decomposition & dec,
			                                                                     vector & distribtuedVector,
																				 size_t v,
																				 size_t p)
	{
		if (distribtuedVector.template getProp<type>(p) == FLUID)
			dec.addComputationCost(v,4);
		else
			dec.addComputationCost(v,3);
	}

	template<typename Decomposition> inline void applyModel(Decomposition & dec, size_t v)
	{
		dec.setSubSubDomainComputationCost(v, dec.getSubSubDomainComputationCost(v) * dec.getSubSubDomainComputationCost(v));
	}

	real_number distributionTol()
	{
		return 1.01;
	}
};

template< typename DistributedParticleVector >
__global__ void
equationOfState_kernel( DistributedParticleVector distributedVector, const RealType& B )
{
	auto i = GET_PARTICLE( distributedVector );

	const RealType rho_i = v( i );
	const RealType rho_frac = rho_a / rho0;
	p( i ) = B * ( std::pow( rho_frac, 7 ) - 1.f );
}

void
equationOfState( DistributedParticleVector& distributedVector )
{
	// particle iterator
	auto distributedParticleVectorGPUIterator = distributedVector.getDomainIteratorGPU();

	CUDA_LAUNCH( equationOfState_kernel,
                distributedParticleVectorGPUIterator,
                distributedVector.toKernel(),
                B );
}


 __device__ __host__ RealType
smoothingFunction_W( const RealType& r, const RealType& h )
{
   const float wConst = 0.02611136f / ( h * h * h ); // 21/(16*PI*h^3)
   const float q = r / h;
   return wConst * ( 1.f + 2.f * q) * ( 2.f - q ) * ( 2.f - q ) * ( 2.f - q ) * ( 2.f - q );
}

__device__ __host__ RealType
smoothingFunction_F( const RealType& r, const RealType& h )
{
   const float wConst =  -0.2611136f / ( h * h * h * h * h) ; // 21/(16*PI*h^5)*(5/8)
   const float q = r / h;
   return wConst * ( 2.f - q ) * ( 2.f - q ) * ( 2.f - q );
}


inline __device__ __host__ RealType
viscousTerm_Pi( const RealType& rhoI, const RealType& rhoJ, const RealType& drs, const RealType& drdv, const RealType& h, const RealType& alpha, const RealType& preventZeroEps )
{
   const RealType mu = h * drdv / ( drs * drs + preventZeroEps );
   return ( drdv < 0.f ) ? ( alpha * mu / ( rhoI + rhoJ ) ) : ( 0.f );
}

template<typename particles_type, typename fluid_ids_type,typename nearestNeighbors_type>
__global__ void
computeFluidInteractions_kernel(particles_type distribtuedVector, fluid_ids_type fids, nearestNeighbors_type nearestNeighbors, real_number W_dap, real_number cbar)
{
	unsigned int a;
	GET_PARTICLE_BY_ID( a, fids );

	// get properties of particle a
	const VectorType r_a = r( a );
	const unsigned int type_a = type( a );
	const RealType rho_a = rho( i );
   const RealType p_a = p( a );
	const VectorType v_a = v( a );

   VectorType dv_dt_a = { 0.f, 0.f, -gravity };
	RealType drho_dt_a = 0.f;
	RealType maxViscosity_a = 0.f;

	// get an iterator over the neighborhood particles of a
	auto neihgborParticlesIterator = nearestNeighbors.getnearestNeighborsIteratorBox( nearestNeighbors.getCell( r_a ) );

	// iterate over particles in neighborhood
	while( neihgborParticlesIterator.isNext() == true )
	{
		const auto b = neihgborPartiicles.get_sort();
      const unsigned int type_b = type( b );
      if( type_a == BOUNDARY && type_b == BOUNDARY || a == b ) //TODO: It boundary/boundary check necessary?
      {
         ++neihgborParticlesIterator;
         continue;
      }

		const VectorType r_b = distributedVector.getPos( b );
      const VectorType v_b = v( b );
      const RealType p_b = p( b );
      const RealType rho_b = rho( b );

		const VectorType r_ab = xa - xb;
		const VectorType v_ab = va - vb;
		const RealType r2 = norm2( dr ); //TODO: Retarded
		const RealType drs = sqrtf( r2 ); //TODO: Duplicit distances

		if ( drs < searchRadius ) //TODO: This condition is too deep
		{
         const VectorType gradW = r_ab * smoothingFunction_F( drs, h );
         const RealType pressureTerm = ( p_a + p_b ) / ( rho_a + rho_b );
         const RealType viscousTerm = viscousTerm_Pi( rho_i, rho_j, drs, ( r_ij, v_ij ), h, alpha, preventZeroEps );

         maxViscosity_a += viscousTerm;
         dv_dt_a += ( -1.f ) * ( pressureTerm + viscousTerm ) * DW  * massb;
			drho_dt_a += ( v_ab, gradW ) * m;
		}

		++neihgborParticlesIterator;
	}
   reductionBufferVisco( a ) = maxViscosity_a;
   dv_dt( a ) = dv_dt_a;
   drho_dt( a ) = drho_dt_a;
}

template<typename particles_type, typename fluid_ids_type,typename nearestNeighbors_type>
__global__ void computeBoundaryInteractions_kernel(particles_type distribtuedVector, fluid_ids_type fbord, nearestNeighbors_type nearestNeighbors, real_number W_dap, real_number cbar)
{
	unsigned int a;
	GET_PARTICLE_BY_ID( a, fbord );

	// get properties of particle a
	const VectorType r_a = r( a );
	const unsigned int type_a = type( a );
	const VectorType v_a = v( a );

	RealType drho_dt_a = 0.f;
	RealType maxViscosity_a = 0.f;

	// get an iterator over the neighborhood particles of a
	auto neihgborParticlesIterator = nearestNeighbors.getnearestNeighborsIteratorBox( nearestNeighbors.getCell( r_a ) );

	// iterate over particles in neighborhood
	while( neihgborParticlesIterator.isNext() == true )
	{
		const auto b = neihgborPartiicles.get_sort();
      const unsigned int type_b = type( b );
      if( type_a == BOUNDARY && type_b == BOUNDARY || a == b ) //TODO: It boundary/boundary check necessary?
      {
         ++neihgborParticlesIterator;
         continue;
      }

		const VectorType r_b = distributedVector.getPos( b );
      const VectorType v_b = v( b );

		const VectorType r_ab = xa - xb;
		const VectorType v_ab = va - vb;
		const RealType r2 = norm2( dr ); //TODO: Retarded
		const RealType drs = sqrtf( r2 ); //TODO: Duplicit distances

		if ( drs < searchRadius ) //TODO: This condition is too deep
		{
         const VectorType gradW = r_ab * smoothingFunction_F( drs, h );
			drho_dt_a += ( v_ab, gradW ) * m;
		}
		++neihgborParticlesIterator;
	}
   reductionBufferVisco( a ) = maxViscosity_a;
   dv_dt( a ) = dv_dt_a;
}

struct ChekFluidType
{
	__device__ static bool check(int c)
	{
		return c == FLUID;
	}
};

struct CheckBoundaryType
{
   __device__ static bool check(int c)
   {
      return c == BOUNDARY;
   }
};

template<typename CellList>
void computeInteractions( DistributedParticleVector& distribtuedVector,
                          CellList & nearestNeighbors,
                          RealType& max_visc,
                          size_t cnt,
                          openfpm::vector_gpu< aggregate< int > >& fluid_ids,
                          openfpm::vector_gpu< aggregate< int > >& border_ids )
{
	// update the cell-list
	distributedVector.updateCellList< PARTICLE_TYPE, RHO, PRESSURE, VELOCITY >( nearestNeighbors );

	// get the particles fluid ids
	get_indexes_by_type< PARTICLE_TYPE, ChekFluidType>( distribtuedVector.getPropVectorSort(),
                                                       fluid_ids,distribtuedVector.size_local(),
                                                       distribtuedVector.getVC().getGpuContext() );

	// get the particles fluid ids
	get_indexes_by_type< PARTICLE_TYPE, CheckBoundaryType >( distribtuedVector.getPropVectorSort(),
                                                            border_ids,distribtuedVector.size_local(),
                                                            distribtuedVector.getVC().getGpuContext() );

	auto part = fluid_ids.getGPUIterator(96);
	CUDA_LAUNCH( computeFluidInteractions_kernel,
                part,distribtuedVector.toKernel_sorted(),
                fluid_ids.toKernel(),
                nearestNeighbors.toKernel(),
                W_dap,
                cbar );

	part = border_ids.getGPUIterator(96);
	CUDA_LAUNCH( computeBoundaryInteractions_kernel,
                part,
                distribtuedVector.toKernel_sorted(),
                border_ids.toKernel(),
                nearestNeighbors.toKernel(),
                W_dap,
                cbar );

	distribtuedVector.merge_sort< DV_DT, DRHO_DT, REDUCTION_REMOVE >( nearestNeighbors );
	max_visc = reduce_local< REDUCTION_REMOVE, _max_ >( distribtuedVector ); //TODO: _max_ is functional? Ugly name.
}

template<typename vector_type>
__global__ void max_acceleration_and_velocity_gpu(vector_type distribtuedVector)
{
	auto a = GET_PARTICLE(distribtuedVector);

	Point<3,real_number> acc(distribtuedVector.template getProp<force>(a));
	distribtuedVector.template getProp<red>(a) = norm(acc);

	Point<3,real_number> vel(distribtuedVector.template getProp<velocity>(a));
	distribtuedVector.template getProp<red2>(a) = norm(vel);
}

void max_acceleration_and_velocity(particles & distribtuedVector, real_number & max_acc, real_number & max_vel)
{
	// Calculate the maximum acceleration
	auto part = distribtuedVector.getDomainIteratorGPU();

	CUDA_LAUNCH(max_acceleration_and_velocity_gpu,part,distribtuedVector.toKernel());

	max_acc = reduce_local<red,_max_>(distribtuedVector);
	max_vel = reduce_local<red2,_max_>(distribtuedVector);

	Vcluster<> & v_cl = create_vcluster();
	v_cl.max(max_acc);
	v_cl.max(max_vel);
	v_cl.execute();
}


real_number calc_deltaT(particles & distribtuedVector, real_number ViscDtMax)
{
	real_number Maxacc = 0.0;
	real_number Maxvel = 0.0;
	max_acceleration_and_velocity(distribtuedVector,Maxacc,Maxvel);

	//-dt1 depends on force per unit mass.
	const real_number dt_f = (Maxacc)?sqrt(H/Maxacc):std::numeric_limits<float>::max();

	//-dt2 combines the Courant and the viscous time-step controls.
	const real_number dt_cv = H/(std::max(cbar,Maxvel*10.f) + H*ViscDtMax);

	//-dt new value of time step.
	real_number dt=real_number(CFLnumber)*std::min(dt_f,dt_cv);
	if(dt<real_number(DtMin))
	{dt=real_number(DtMin);}

	return dt;
}

template< typename DistributedParticleVector >
__global__ void
verletIntegrationScheme_kernel( DistributedParticleVector distributedVector, const RealType dt ) //, real_number dt2, real_number dt205)
{
	auto i = GET_PARTICLE( distributedVector );

	// if the particle is boundary, update density
	if ( type( i ) == ParticleTypes::Wall )
	{
      v( i ) = 0.f;

      const RealType = rhoBackup_i = rho( i );
      const RealType = rhoNew_i = rho_old( i ) + dt2 * drho_dt( i );
    	rho( i ) = ( rhoNew_i < rho0 ) ? rho0 : rhoNew_i ;
	   rho_old( i ) = rhoBackup_i;

	   reductionBufferRemove( i ) = 0;
		return;
	}

	// if the particle is fluid, update position and density
   const VectorType dr_i = dt * v( i ) + dt205 * dv_dt( i );
   r( i ) += dr_i;

   const VectorType vToBackup_i = v( i );
   v( i ) = v_old( i ) + dt2 * dv_dt( i );

   const RealType rhoToBackup_i = rho( i );
   rho( i ) = rho_old( i ) + dt2 * drho_dt;

   // check if the particle go out of range in space and in density
   const VectorType r_i = r( i );
   const RealType rho_i = rho( i );
   if ( r_i[ 0 ] < 0.0  || r_i[ 1 ] < 0.0 || r_i[ 2 ] < 0.0 ||
        r_i[ 0 ] > 3.22 || r_i[ 1 ] > 1.0 || r_i[ 2 ] > 1.5 ||
		  rho_i < RhoMin  || rho_i > RhoMax )
    {
    	reductionBufferRemove( i ) = 1;
    }
    else
    {
    	reductionBufferRemove( i ) = 0;
    }

    v_old( i ) = vToBackup_i;
    rho_old( i ) = rhoToBackup_i;
}

size_t cnt = 0;

void
verletIntegrationScheme( DistributedParticleVector& distributedVector, const RealType& dt )
{
	// particle iterator
	auto distributedParticleVectorGPUIterator = distributedVector.getDomainIteratorGPU();

	const RealType dt205 = dt * dt * 0.5;
	const RealType dt2 = dt * 2.0;

	CUDA_LAUNCH( verletIntegrationScheme_kernel,
                distributedParticleVectorGPUIterator,
                distributedVector.toKernel(),
                dt,
                dt2,
                dt205 );

	// remove the marked particles
	remove_marked< REDUCTION_REMOVE >( distributedVector );

	// increment the iteration counter
	cnt++;
}

template<typename DistributedParticleVector >
__global__ void
eulerIntegrationScheme_kernel( DistributedParticleVector& DistributedParticleVector, const RealType dt, const RealType dt205 )
{
	auto i = GET_PARTICLE( distributedVector );

	// if the particle is boundary, update density
	if ( type( i ) == ParticleTypes::Wall )
	{
      v( i ) = 0.f;

      const RealType = rhoBackup_i = rho( i );
      const RealType = rhoNew_i = rho( i ) + dt * drho_dt( i );
    	rho( i ) = ( rhoNew_i < rho0 ) ? rho0 : rhoNew_i ;
	   rho_old( i ) = rhoBackup_i;

	   reductionBufferRemove( i ) = 0;
		return;
	}

	// if the particle is fluid, update position and density
   const VectorType dr_i = dt * v( i ) + dt205 * dv_dt( i );
   r( i ) += dr_i;

   const VectorType vToBackup_i = v( i );
   v( i ) = v( i ) + dt * dv_dt( i );

   const RealType rhoToBackup_i = rho( i );
   rho( i ) = rho( i ) + dt * drho_dt( i );

   // check if the particle go out of range in space and in density
   const VectorType r_i = r( i );
   const RealType rho_i = rho( i );
   if ( r_i[ 0 ] < 0.0  || r_i[ 1 ] < 0.0 || r_i[ 2 ] < 0.0 ||
        r_i[ 0 ] > 3.22 || r_i[ 1 ] > 1.0 || r_i[ 2 ] > 1.5 ||
		  rho_i < RhoMin  || rho_i > RhoMax )
    {
    	reductionBufferRemove( i ) = 1;
    }
    else
    {
    	reductionBufferRemove( i ) = 0;
    }

    v_old( i ) = vToBackup_i;
    rho_old( i ) = rhoToBackup_i;
}

void
eulerIntegrationScheme( DistributedParticleVector& distributedVector, const RealType& dt )
{
	// particle iterator
	auto distributedParticleVectorGPUIterator = distributedVector.getDomainIteratorGPU();

	const RealType dt205 = dt * dt * 0.5;

	CUDA_LAUNCH( eulerIntegrationScheme_kernel,
                distributedParticleVectorGPUIterator,
                DistributedParticleVector.toKernel(),
                dt,
                dt205 );

	// remove the particles
	remove_marked< REDUCTION_REMOVE >( distributedVector );

	// increment the iteration counter
	cnt++;
}

template<typename vector_type, typename nearestNeighbors_type>
__global__ void sensor_pressure_gpu(vector_type distribtuedVector, nearestNeighbors_type nearestNeighbors, Point<3,real_number> probe, real_number * press_tmp)
{
	real_number tot_ker = 0.0;

	// Get the position of the probe i
	Point<3,real_number> xp = probe;

	// get the iterator over the neighbohood particles of the probes position
	auto itg = nearestNeighbors.getnearestNeighborsIteratorBox(nearestNeighbors.getCell(xp));
	while (itg.isNext())
	{
		auto q = itg.get_sort();

		// Only the fluid particles are importants
		if (distribtuedVector.template getProp<type>(q) != FLUID)
		{
			++itg;
			continue;
		}

		// Get the position of the neighborhood particle q
		Point<3,real_number> xq = distribtuedVector.getPos(q);

		// Calculate the contribution of the particle to the pressure
		// of the probe
		real_number r = sqrt(norm2(xp - xq));

		real_number ker = Wab(r) * (MassFluid / rho_zero);

		// Also keep track of the calculation of the summed
		// kernel
		tot_ker += ker;

		// Add the total pressure contribution
		*press_tmp += distribtuedVector.template getProp<Pressure>(q) * ker;

		// next neighborhood particle
		++itg;
	}

	// We calculate the pressure normalizing the
	// sum over all kernels
	if (tot_ker == 0.0)
	{*press_tmp = 0.0;}
	else
	{*press_tmp = 1.0 / tot_ker * *press_tmp;}
}

template<typename Vector, typename CellList>
inline void sensor_pressure(Vector & distribtuedVector,
                            CellList & nearestNeighbors,
                            openfpm::vector<openfpm::vector<real_number>> & press_t,
                            openfpm::vector<Point<3,real_number>> & probes)
{
    Vcluster<> & v_cl = create_vcluster();

    press_t.add();

    for (size_t i = 0 ; i < probes.size() ; i++)
    {
    	// A float variable to calculate the pressure of the problem
    	CudaMemory press_tmp_(sizeof(real_number));
    	real_number press_tmp;

        // if the probe is inside the processor domain
		if (distribtuedVector.getDecomposition().isLocal(probes.get(i)) == true)
		{
			CUDA_LAUNCH_DIM3(sensor_pressure_gpu,1,1,distribtuedVector.toKernel_sorted(),nearestNeighbors.toKernel(),probes.get(i),(real_number *)press_tmp_.toKernel());

			//distribtuedVector.merge<Pressure>(nearestNeighbors);

			// move calculated pressure on
			press_tmp_.deviceToHost();
			press_tmp = *(real_number *)press_tmp_.getPointer();
		}

		// This is not necessary in principle, but if you
		// want to make all processor aware of the history of the calculated
		// pressure we have to execute this
		v_cl.sum(press_tmp);
		v_cl.execute();

		// We add the calculated pressure into the history
		press_t.last().add(press_tmp);
	}
}

template<typename vector_type, typename nearestNeighbors_type>
__global__ void sensor_water_level_gpu(vector_type distribtuedVector, nearestNeighbors_type nearestNeighbors, Point<3,real_number> probe, real_number * wl_tmp, const size_t number_of_levels_to_test)
{
   unsigned int p = blockDim.x*blockIdx.x + threadIdx.x;
   if( p >= number_of_levels_to_test )
      return;

	real_number tot_ker = 0.0;

	// Get the position of the probe i
   Point<3,real_number> level = {0., 0., p * dp};
	Point<3,real_number> xp = probe + level;

	// get the iterator over the neighbohood particles of the probes position
	auto itg = nearestNeighbors.getnearestNeighborsIteratorBox(nearestNeighbors.getCell(xp));
	while (itg.isNext())
	{
		auto q = itg.get_sort();

		// Only the fluid particles are important
		//if (distribtuedVector.template getProp<type>(q) != FLUID)
		//{
		//	++itg;
		//	continue;
		//}

		// Get the position of the neighborhood particle q
		Point<3,real_number> xq = distribtuedVector.getPos(q);

		// Calculate the kernel occupancy
		real_number r = sqrt(norm2(xp - xq));
		real_number ker = Wab(r) * (MassFluid / distribtuedVector.template getProp<rho>(q));
		tot_ker += ker;

		// next neighborhood particle
		++itg;
      //if( ker != 0.0 )
      //   printf(" << tot_ker, Wab, MassFluid, rho >>: [%f , %f, %f, %f]", tot_ker, ker, MassFluid, distribtuedVector.template getProp<rho>(q) );

	}
   wl_tmp[p] = tot_ker;
   //printf(" tot_ker: %f ", tot_ker);
}

template<typename Vector, typename CellList>
inline void sensor_water_level(Vector & distribtuedVector,
                               CellList & nearestNeighbors,
                               openfpm::vector<openfpm::vector<real_number>> & water_level_t,
                               openfpm::vector<Point<3,real_number>> & probes_water_level)
{
    Vcluster<> & v_cl = create_vcluster();

    water_level_t.add();

    for (size_t i = 0 ; i < probes_water_level.size() ; i++)
    {
      //sampled water levels
      openfpm::vector<real_number> tested_water_levels;
      size_t number_of_levels_to_test = (size_t) std::floor(1.5/dp); //TODO: approx. box height

    	//A float variable to calculate the pressure of the problem
    	CudaMemory wl_tmp_(number_of_levels_to_test * sizeof(real_number));
    	real_number *wl_tmp;

      //if the probe is inside the processor domain
		if (distribtuedVector.getDecomposition().isLocal(probes_water_level.get(i)) == true)
		{
			CUDA_LAUNCH_DIM3(sensor_water_level_gpu,1,number_of_levels_to_test,distribtuedVector.toKernel_sorted(),nearestNeighbors.toKernel(),probes_water_level.get(i),(real_number *)wl_tmp_.toKernel(), number_of_levels_to_test);

			//distribtuedVector.merge<Pressure>(nearestNeighbors);

			// move calculated pressure on
			wl_tmp_.deviceToHost();
			//wl_tmp = *(real_number *)wl_tmp_.getPointer();
         wl_tmp = static_cast<real_number*>(const_cast<void*>(wl_tmp_.getPointer()));
		}

      //Obtain the actual water level from the samples
      real_number water_level_temp = 0;
      //std::cout << "senzor: " << i << std::endl;
      //for( size_t j = 1 ; j < number_of_levels_to_test ; j ++)
      //{
      //   std::cout << wl_tmp[j] << " ";
      //   //if( wl_tmp[j] < 0.5f ){
      //   //   water_level_temp = j * dp;
      //   //   break;
      //   //}
      //}
      std::cout << std::endl;
      for( size_t j = 1 ; j < number_of_levels_to_test - 1 ; j ++)
      {
         if( ( wl_tmp[j] < 0.5f ) && ( wl_tmp[j+1] < 0.5f ) ){
            water_level_temp = j * dp;
            break;
         }
      }
      std::cout << "Water level temp = " << water_level_temp << std::endl;

		// This is not necessary in principle, but if you
		// want to make all processor aware of the history of the calculated
		// pressure we have to execute this
		v_cl.sum(water_level_temp);
		v_cl.execute();

		// We add the calculated pressure into the history
		water_level_t.last().add(water_level_temp);
	}
}

int main(int argc, char* argv[])
{
    // initialize the library
	openfpm_init(&argc,&argv);

	openfpm::vector_gpu<aggregate<int>> fluid_ids;
	openfpm::vector_gpu<aggregate<int>> border_ids;

#ifdef CUDIFY_USE_CUDA
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
#endif

	// It contain for each time-step the value detected by the probes
	openfpm::vector<openfpm::vector<real_number>> press_t;
	openfpm::vector<Point<3,real_number>> probes;
   std::vector<real_number> press_measured_times;

	openfpm::vector<openfpm::vector<real_number>> water_level_t;
	openfpm::vector<Point<3,real_number>> probes_water_level;

	probes.add({0.8245,0.471,0.021});
	probes.add({0.8245,0.471,0.061});
	probes.add({0.8245,0.471,0.101});
	probes.add({0.8245,0.471,0.141});
	probes.add({0.8035,0.471,0.1645});
	probes.add({0.7635,0.471,0.1645});
	probes.add({0.7235,0.471,0.1645});
	probes.add({0.6835,0.471,0.1645});

   probes_water_level.add({0.496,0.5,0.0});
   probes_water_level.add({0.992,0.5,0.0});
   probes_water_level.add({1.488,0.5,0.0});
   probes_water_level.add({2.638,0.5,0.0});

	// Here we define our domain a 2D box with internals from 0 to 1.0 for x and y
	//Box<3,real_number> domain({-0.05,-0.05,-0.05},{1.7010,0.7065,0.511});
	//size_t sz[3] = {413,179,133};
	Box<3,real_number> domain({-0.05,-0.05,-0.05},{3.3210,1.1065,1.511});
   //TOOD: Parametrize this
   size_t sz_x = ( size_t ) std::ceil((3.3210 + 0.05)/dp);
   size_t sz_y = ( size_t ) std::ceil((1.1065 + 0.05)/dp);
   size_t sz_z = ( size_t ) std::ceil((1.511 + 0.05)/dp);
	size_t sz[3] = {sz_x,sz_y,sz_z};

	// Fill W_dap
	W_dap = 1.0/Wab(H/1.5);

	// Here we define the boundary conditions of our problem
    size_t bc[3]={NON_PERIODIC,NON_PERIODIC,NON_PERIODIC};

	// extended boundary around the domain, and the processor domain
	Ghost<3,real_number> g(2*H);

	particles distribtuedVector(0,domain,bc,g,DEC_GRAN(128));

	//! \cond [draw fluid] \endcond

	// You can ignore all these dp/2.0 is a trick to reach the same initialization
	// of Dual-SPH that use a different criteria to draw particles
	//Box<3,real_number> fluid_box({dp/2.0,dp/2.0,dp/2.0},{0.4+dp/2.0,0.67-dp/2.0,0.3+dp/2.0});
	Box<3,real_number> fluid_box({1.992 + dp/2.0,dp/2.0,dp/2.0},{3.22-dp/2.0,1.0-dp/2.0,0.55+dp/2.0});

	// return an iterator to the fluid particles to add to distribtuedVector
	auto fluid_it = DrawParticles::DrawBox(distribtuedVector,sz,domain,fluid_box);

	// here we fill some of the constants needed by the simulation
	max_fluid_height = fluid_it.getBoxMargins().getHigh(2);
	h_swl = fluid_it.getBoxMargins().getHigh(2) - fluid_it.getBoxMargins().getLow(2);
	B = (coeff_sound)*(coeff_sound)*gravity*h_swl*rho_zero / gamma_;
	cbar = coeff_sound * sqrt(gravity * h_swl);

	// for each particle inside the fluid box ...
	while (fluid_it.isNext())
	{
		// ... add a particle ...
		distribtuedVector.add();

		// ... and set it position ...
		distribtuedVector.getLastPos()[0] = fluid_it.get().get(0);
		distribtuedVector.getLastPos()[1] = fluid_it.get().get(1);
		distribtuedVector.getLastPos()[2] = fluid_it.get().get(2);

		// and its type.
		distribtuedVector.template getLastProp<type>() = FLUID;

		// We also initialize the density of the particle and the hydro-static pressure given by
		//
		// rho_zero*g*h = P
		//
		// rho_p = (P/B + 1)^(1/Gamma) * rho_zero
		//

		distribtuedVector.template getLastProp<Pressure>() = rho_zero * gravity *  (max_fluid_height - fluid_it.get().get(2));

		distribtuedVector.template getLastProp<rho>() = pow(distribtuedVector.template getLastProp<Pressure>() / B + 1, 1.0/gamma_) * rho_zero;
		distribtuedVector.template getLastProp<rho_prev>() = distribtuedVector.template getLastProp<rho>();
		distribtuedVector.template getLastProp<velocity>()[0] = 0.0;
		distribtuedVector.template getLastProp<velocity>()[1] = 0.0;
		distribtuedVector.template getLastProp<velocity>()[2] = 0.0;

		distribtuedVector.template getLastProp<velocity_prev>()[0] = 0.0;
		distribtuedVector.template getLastProp<velocity_prev>()[1] = 0.0;
		distribtuedVector.template getLastProp<velocity_prev>()[2] = 0.0;

		// next fluid particle
		++fluid_it;
	}

	// Recipient
	//Box<3,real_number> recipient1({0.0,0.0,0.0},{1.6+dp/2.0,0.67+dp/2.0,0.4+dp/2.0});
	//Box<3,real_number> recipient2({dp,dp,dp},{1.6-dp/2.0,0.67-dp/2.0,0.4+dp/2.0});
	Box<3,real_number> recipient1({0.0,0.0,0.0},{3.22+dp/2.0,1.0+dp/2.0,1.0+dp/2.0});
	Box<3,real_number> recipient2({dp,dp,dp},{3.22-dp/2.0,1.0-dp/2.0,1.0+dp/2.0});

	//Box<3,real_number> obstacle1({0.9,0.24-dp/2.0,0.0},{1.02+dp/2.0,0.36,0.45+dp/2.0});
	//Box<3,real_number> obstacle2({0.9+dp,0.24+dp/2.0,0.0},{1.02-dp/2.0,0.36-dp,0.45-dp/2.0});
	//Box<3,real_number> obstacle3({0.9+dp,0.24,0.0},{1.02,0.36,0.45});

	Box<3,real_number> obstacle1({0.66,0.3-dp/2.0,0.0},{0.82+dp/2.0,0.7,0.16+dp/2.0});
	Box<3,real_number> obstacle2({0.66+dp,0.3+dp/2.0,0.0},{0.82-dp/2.0,0.7-dp,0.16-dp/2.0});
	Box<3,real_number> obstacle3({0.66+dp,0.3,0.0},{1.02,0.7,0.16});

	openfpm::vector<Box<3,real_number>> holes;
	holes.add(recipient2);
	holes.add(obstacle1);
	auto bound_box = DrawParticles::DrawSkin(distribtuedVector,sz,domain,holes,recipient1);

	while (bound_box.isNext())
	{
		distribtuedVector.add();

		distribtuedVector.getLastPos()[0] = bound_box.get().get(0);
		distribtuedVector.getLastPos()[1] = bound_box.get().get(1);
		distribtuedVector.getLastPos()[2] = bound_box.get().get(2);

		distribtuedVector.template getLastProp<type>() = BOUNDARY;
		distribtuedVector.template getLastProp<rho>() = rho_zero;
		distribtuedVector.template getLastProp<rho_prev>() = rho_zero;
		distribtuedVector.template getLastProp<velocity>()[0] = 0.0;
		distribtuedVector.template getLastProp<velocity>()[1] = 0.0;
		distribtuedVector.template getLastProp<velocity>()[2] = 0.0;

		distribtuedVector.template getLastProp<velocity_prev>()[0] = 0.0;
		distribtuedVector.template getLastProp<velocity_prev>()[1] = 0.0;
		distribtuedVector.template getLastProp<velocity_prev>()[2] = 0.0;

		++bound_box;
	}

	auto obstacle_box = DrawParticles::DrawSkin(distribtuedVector,sz,domain,obstacle2,obstacle1);

	while (obstacle_box.isNext())
	{
		distribtuedVector.add();

		distribtuedVector.getLastPos()[0] = obstacle_box.get().get(0);
		distribtuedVector.getLastPos()[1] = obstacle_box.get().get(1);
		distribtuedVector.getLastPos()[2] = obstacle_box.get().get(2);

		distribtuedVector.template getLastProp<type>() = BOUNDARY;
		distribtuedVector.template getLastProp<rho>() = rho_zero;
		distribtuedVector.template getLastProp<rho_prev>() = rho_zero;
		distribtuedVector.template getLastProp<velocity>()[0] = 0.0;
		distribtuedVector.template getLastProp<velocity>()[1] = 0.0;
		distribtuedVector.template getLastProp<velocity>()[2] = 0.0;

		distribtuedVector.template getLastProp<velocity_prev>()[0] = 0.0;
		distribtuedVector.template getLastProp<velocity_prev>()[1] = 0.0;
		distribtuedVector.template getLastProp<velocity_prev>()[2] = 0.0;

		++obstacle_box;
	}

	distribtuedVector.map();

	// Now that we fill the vector with particles
	ModelCustom md;

	distribtuedVector.addComputationCosts(md);
	distribtuedVector.getDecomposition().decompose();
	distribtuedVector.map();

	///////////////////////////

	// Ok the initialization is done on CPU on GPU we are doing the main loop, so first we offload all properties on GPU

	distribtuedVector.hostToDevicePos();
	distribtuedVector.template hostToDeviceProp<type,rho,rho_prev,Pressure,velocity>();


	distribtuedVector.ghost_get<type,rho,Pressure,velocity>(RUN_ON_DEVICE);

	auto nearestNeighbors = distribtuedVector.getCellListGPU/*<CELLLIST_GPU_SPARSE<3,float>>*/(2*H / 2.0);
	nearestNeighbors.setBoxnearestNeighbors(2);


   //Added timers to track every operation inside the time loop
	timer tot_sim;

   timer timer_vcluster;
   float vcluster_total_time = 0.f ;

   timer timer_interaction;
   float interaction_total_time = 0.f ;

   timer tot_pressure;
   float pressure_total_time = 0.f ;

   timer timer_integrate;
   float integration_total_time = 0.f;

   timer tot_rebalancing;
   float rebalanting_total_time = 0.f;

   timer timer_map;
   float map_total_time = 0.f;

   timer timer_reductions;
   float reduction_total_time = 0.f;

   timer timer_ghosts;
   float ghost_total_time = 0.f;

   //Cout the total number of steps
   int tot_steps = 0;

	tot_sim.start();

	size_t write = 0;
	float write_pressure = 0;
	float write_pressure_interval = 0.01;
	size_t it = 0;
	size_t it_reb = 0;
	real_number t = 0.0;
	while (t <= t_end)
	{
      tot_steps++;

      timer_vcluster.start();
		Vcluster<> & v_cl = create_vcluster();
      timer_vcluster.stop();
      vcluster_total_time += timer_vcluster.getwct();
		timer it_time;
		it_time.start();

		////// Do rebalancing every 200 timesteps
      tot_rebalancing.start();
		it_reb++;
		if (it_reb == 300)
		{
			distribtuedVector.map(RUN_ON_DEVICE);

			// Rebalancer for now work on CPU , so move to CPU
            distribtuedVector.deviceToHostPos();
            distribtuedVector.template deviceToHostProp<type>();

			it_reb = 0;
			ModelCustom md;
			distribtuedVector.addComputationCosts(md);
			distribtuedVector.getDecomposition().decompose();

			if (v_cl.getProcessUnitID() == 0)
			{std::cout << "REBALANCED " << it_reb << std::endl;}
		}
      tot_rebalancing.stop();
      rebalanting_total_time += tot_rebalancing.getwct();

      timer_map.start();
		distribtuedVector.map(RUN_ON_DEVICE);
      timer_map.stop();
      rebalanting_total_time += timer_map.getwct();

		// Calculate pressure from the density
      tot_pressure.start();
		equationOfState(distribtuedVector);
      tot_pressure.stop();
      pressure_total_time += tot_pressure.getwct();

		real_number max_visc = 0.0;

      timer_ghosts.start();
		distribtuedVector.ghost_get<type,rho,Pressure,velocity>(RUN_ON_DEVICE);
      timer_ghosts.stop();
      ghost_total_time += timer_ghosts.getwct();


		// Calc forces
      timer_interaction.start();
		computeInteractions(distribtuedVector,nearestNeighbors,max_visc,cnt,fluid_ids,border_ids);
      timer_interaction.stop();
      interaction_total_time += timer_interaction.getwct();

      timer_reductions.start();
		// Get the maximum viscosity term across processors
		v_cl.max(max_visc);
		v_cl.execute();

		// Calculate delta t integration
		real_number dt = calc_deltaT(distribtuedVector,max_visc);

      timer_reductions.stop();
      reduction_total_time += timer_reductions.getwct();

		// VerletStep or euler step
      timer_integrate.start();
		it++;
		if (it < 40)
			verletIntegrationScheme(distribtuedVector,dt);
		else
		{
			eulerIntegrationScheme(distribtuedVector,dt);
			it = 0;
		}
      timer_integrate.stop();
      integration_total_time += timer_integrate.getwct();

		t += dt;

      if( write_pressure <= t )
      {
			// Sensor pressure require update ghost, so we ensure that particles are distributed correctly
			// and ghost are updated
			distribtuedVector.map(RUN_ON_DEVICE);
			distribtuedVector.ghost_get<type,rho,Pressure,velocity>(RUN_ON_DEVICE);
			distribtuedVector.updateCellList(nearestNeighbors);

			// calculate the pressure at the sensor points
			sensor_pressure(distribtuedVector,nearestNeighbors,press_t,probes);
			sensor_water_level(distribtuedVector,nearestNeighbors,water_level_t,probes_water_level);
         press_measured_times.push_back(t);

         write_pressure += write_pressure_interval;
      }

		if (write < t*10)
		{
			// Sensor pressure require update ghost, so we ensure that particles are distributed correctly
			// and ghost are updated (NOTE: not sure, if this is necessary for the output aswell)
			distribtuedVector.map(RUN_ON_DEVICE);
			distribtuedVector.ghost_get<type,rho,Pressure,velocity>(RUN_ON_DEVICE);
			distribtuedVector.updateCellList(nearestNeighbors);

			std::cout << "OUTPUT " << dt << std::endl;

			// When we write we have move all the particles information back to CPU

			distribtuedVector.deviceToHostPos();
			distribtuedVector.deviceToHostProp<type,rho,rho_prev,Pressure,drho,force,velocity,velocity_prev,red,red2>();

			// We copy on another vector with less properties to reduce the size of the output
			vector_dist_gpu<3,real_number,aggregate<unsigned int,real_number[3]>> distribtuedVector_out(distribtuedVector.getDecomposition(),0);

			auto ito = distribtuedVector.getDomainIterator();

			while(ito.isNext())
			{
				auto p = ito.get();

				distribtuedVector_out.add();

				distribtuedVector_out.getLastPos()[0] = distribtuedVector.getPos(p)[0];
				distribtuedVector_out.getLastPos()[1] = distribtuedVector.getPos(p)[1];
				distribtuedVector_out.getLastPos()[2] = distribtuedVector.getPos(p)[2];

				distribtuedVector_out.template getLastProp<0>() = distribtuedVector.template getProp<type>(p);

				distribtuedVector_out.template getLastProp<1>()[0] = distribtuedVector.template getProp<velocity>(p)[0];
				distribtuedVector_out.template getLastProp<1>()[1] = distribtuedVector.template getProp<velocity>(p)[1];
				distribtuedVector_out.template getLastProp<1>()[2] = distribtuedVector.template getProp<velocity>(p)[2];

				++ito;
			}

			distribtuedVector_out.write_frame("Particles",write,VTK_WRITER | FORMAT_BINARY);
			write++;

			if (v_cl.getProcessUnitID() == 0)
			{std::cout << "TIME: " << t << "  write " << it_time.getwct() << "   " << it_reb << "   " << cnt << " Max visc: " << max_visc << "   " << distribtuedVector.size_local()  << std::endl;}
		}

		if (v_cl.getProcessUnitID() == 0)
		{std::cout << "TIME: " << t << "  " << it_time.getwct() << "   " << it_reb << "   " << cnt  << " Max visc: " << max_visc << "   " << distribtuedVector.size_local() << std::endl;}
	}

	tot_sim.stop();

   std::cout << "TIME MEASUREMENT RESULTS:" << std::endl;
	std::cout << "Time to complete: "   << tot_sim.getwct()           << " seconds" << std::endl;
	std::cout << "Vcluster: "           << vcluster_total_time        << " seconds" << std::endl;
	std::cout << "Interaction: "        << interaction_total_time     << " seconds" << std::endl;
	std::cout << "Pressure: "           << pressure_total_time        << " seconds" << std::endl;
	std::cout << "Integration: "        << integration_total_time     << " seconds" << std::endl;
	std::cout << "Reabalancing: "       << rebalanting_total_time     << " seconds" << std::endl;
	std::cout << "Map: "                << map_total_time             << " seconds" << std::endl;
	std::cout << "Reduction: "          << reduction_total_time       << " seconds" << std::endl;
	std::cout << "Ghost: "              << ghost_total_time           << " seconds" << std::endl;
	std::cout << "Number of steps: "    << tot_steps                  << " seconds" << std::endl;

   //Write pressure sensor outputs
   std::ofstream file_probes;
   file_probes.open ("pressure_sensors.csv");
   for (size_t i = 0 ; i < press_measured_times.size() ; i++)
   {
      file_probes << press_measured_times[ i ] ;
      for (size_t j = 0 ; j < probes.size() ; j++){
         file_probes << " " << press_t.get(i).get(j);
      }
      file_probes << std::endl;
   }
   file_probes.close();

   //Write pressure sensor outputs
   std::ofstream file_probes_wl;
   file_probes_wl.open ("pressure_waterLevel.csv");
   for (size_t i = 0 ; i < press_measured_times.size() ; i++)
   {
      file_probes_wl << press_measured_times[ i ] ;
      for (size_t j = 0 ; j < probes_water_level.size() ; j++){
         file_probes_wl << " " << water_level_t.get(i).get(j);
      }
      file_probes_wl << std::endl;
   }
   file_probes_wl.close();

   //Write timer into json like structure
   std::ofstream file_timers;
   file_timers.open ("timers.json");

   file_timers <<"{" << std::endl;
   file_timers <<"	\"integrate\": \"" << integration_total_time  << "\"," << std::endl;
   file_timers <<"	\"integrate-average\": \"" << integration_total_time / tot_steps << "\"," << std::endl;
   file_timers <<"	\"interaction\": \"" << interaction_total_time  << "\"," << std::endl;
   file_timers <<"	\"interaction-average\": \"" << interaction_total_time / tot_steps << "\"," << std::endl;
   file_timers <<"	\"pressure-update\": \"" << pressure_total_time  << "\"," << std::endl;
   file_timers <<"	\"pressure-update-average\": \"" << pressure_total_time / tot_steps << "\"," << std::endl;
   file_timers <<"	\"vcluster\": \"" << vcluster_total_time << "\"," << std::endl;
   file_timers <<"	\"vcluster-average\": \"" << vcluster_total_time / tot_steps << "\"," << std::endl;
   file_timers <<"	\"rebalancing\": \"" << rebalanting_total_time << "\"," << std::endl;
   file_timers <<"	\"rebalancing-average\": \"" << rebalanting_total_time / tot_steps << "\"," << std::endl;
   file_timers <<"	\"map\": \"" << map_total_time << "\"," << std::endl;
   file_timers <<"	\"map-average\": \"" << map_total_time / tot_steps << "\"," << std::endl;
   file_timers <<"	\"reduction\": \"" << reduction_total_time << "\"," << std::endl;
   file_timers <<"	\"reduction_total_time-average\": \"" << reduction_total_time / tot_steps << "\"," << std::endl;
   file_timers <<"	\"ghost\": \"" << ghost_total_time << "\"," << std::endl;
   file_timers <<"	\"ghost-average\": \"" << ghost_total_time / tot_steps << "\"," << std::endl;
   file_timers <<"	\"total\": \"" << tot_sim.getwct()  << "\"," << std::endl;
   file_timers <<"	\"total-average\": \"" << tot_sim.getwct() / tot_steps << "\"" << std::endl;
   file_timers <<"}" << std::endl;

   file_timers.close();


	openfpm_finalize();
}

#else

int main(int argc, char* argv[])
{
        return 0;
}

#endif
