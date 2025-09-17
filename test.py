import numpy as np
import random
from ENV import uavENV

def run_smoke_test(n_uavs, n_ues, steps, seed):
    np.random.seed(seed)
    random.seed(seed)

    env = uavENV(n_uavs=n_uavs, n_ues=n_ues)
    obs = env.reset()

    print("Reset done.")
    print("Observation per-agent shape:", obs.shape)   # expect (n_uavs, obs_dim)
    print("Initial UE tasks (bits):", env.ue_tasks)
    print("Initial UAV battery:", env.uav_battery)
    print()

    for step in range(steps):
        # sample random continuous actions in [-1, 1] for each UAV
        actions = np.random.uniform(-1.0, 1.0, size=(n_uavs, 4)).astype(np.float32)
        print("actions:", actions)
        obs, rewards, done, info = env.step(actions)

        print(f"Step {step+1}/{steps}")
        print("  rewards:", rewards)
        print("  any tasks finished this step:", info["task_completed"].any())
        # show remaining total task bits and mean battery
        print("  total remaining UE tasks (sum bits):", np.sum(info["ue_tasks"]))
        print("  per-UAV battery:", info["uav_battery"])
        print("  done:", done)
        print(obs)

        if not done:
            print("Environment signaled done at step", step+1)
            break

    print("Final UE tasks:", env.ue_tasks)
    print("Final UAV battery:", env.uav_battery)

if __name__ == "__main__":
    run_smoke_test(n_uavs=4, n_ues=2, steps=20, seed=123)