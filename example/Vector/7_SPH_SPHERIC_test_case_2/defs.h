// define basic simulation types
using RealType = float;
using IndexType = int;
using VectorType = TNL::Containers::StaticVector< 3, RealType >;

// define used fileds
using AggregatedVariables = aggregate< unsigned int,     // particle type
                                       RealType,         // density
                                       RealType,         // densiy at n - 1
                                       RealType,         // pressure
                                       RealType,         // time derivative of density
                                       VectorType,       // time derivative of velocity
                                       VectorType,       // velocity
                                       VectorType,       // velocity n - 1
                                       RealType,         // time step reduction buffer
                                       RealType >;       // viscosity reduction buffer

// define type of vecor to store the data
using DistributedParticleVector =  vector_dist_gpu< 3, RealType, AggregatedVariables >;

#define PARTICLE_TYPE 0
#define DRHO_DT 4
#define DV_DT 5
#define REDUCTION_REMOVE 8
#define REDUCTION_VISCO 9


// define more friendly acces to variables
//#define r( i ) distributedVector.getPos( i )
#define r( i ) { distributedVector.getPos( i )[ 0 ], distributedVector.getPos( i )[ 1 ], distributedVector.getPos( i )[ 2 ] }
#define type( i ) distributedVector.template getProp< 0 >( i )
#define rho( i ) distributedVector.template getProp< 1 >( i )
#define rho_old( i ) distributedVector.template getProp< 2 >( i )
#define p( i ) distributedVector.template getProp< 3 >( i )
#define drho_dt( i ) distributedVector.template getProp< 4 >( i )
#define dv_dt( i ) distributedVector.template getProp< 5 >( i )
#define v( i ) distributedVector.template getProp< 6 >( i )
#define v_old( i ) distributedVector.template getProp< 7 >( i )
#define reductionBufferRemove( i ) distributedVector.template getProp< 8 >( i )
#define reductionBufferVisco( i ) distributedVector.template getProp< 9 >( i )


// define used types of particles
enum class ParticleTypes
: std::uint8_t
{
    Fluid = 0,
    Wall = 1,
    Inlet = 2,
    Outlet = 3
};

