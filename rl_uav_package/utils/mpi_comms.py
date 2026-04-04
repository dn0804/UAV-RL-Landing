"""
MPI communication primitives for parallel PPO training.

Handles weight synchronization and curriculum state broadcast
between Rank 0 (learner) and worker ranks.
"""

import pickle


def broadcast_weights(model, comm, rank):
    """Rank 0 broadcasts policy state_dict to all workers.

    Uses pickle serialization (lowercase bcast).  The policy is
    ~200KB so serialization overhead is negligible.

    Parameters
    ----------
    model : SafePPO
        The PPO model (all ranks have a local copy).
    comm : MPI.Comm
        MPI communicator.
    rank : int
        This process's MPI rank.
    """
    if rank == 0:
        data = pickle.dumps(model.policy.state_dict())
    else:
        data = None
    data = comm.bcast(data, root=0)
    if rank != 0:
        model.policy.load_state_dict(pickle.loads(data))


def broadcast_curriculum(curriculum_state, comm):
    """Broadcast curriculum state tuple from Rank 0 to all workers.

    Parameters
    ----------
    curriculum_state : tuple or None
        (stage, is_blending, blend_step) on Rank 0, None on workers.
    comm : MPI.Comm
        MPI communicator.

    Returns
    -------
    curriculum_state : tuple
        The broadcast state on all ranks.
    """
    return comm.bcast(curriculum_state, root=0)
