import numpy as np
from envs.rov_dyn_env import ROVDynEnv

def main():
    env = ROVDynEnv(
        dt=0.02,
        max_power=60.0,
        device='cpu',
        window_size=9,
        w_err=1.0,
        w_jerk=0.01,
        w_eng=0.001,
        accel_filter_alpha=0.5,
    )
    obs, _ = env.reset(seed=0)
    print("Initial observation:", obs)

    for t in range(100):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        if t % 10 == 0:
            print(f"Step {t:03d}: pos={info['pos']}, vel={info['vel']}, accel={info['accel']}, reward={reward:.3f}")
        if done:
            break

    env.close()
    print("Random rollout test completed successfully.")

if __name__ == "__main__":
    main()
