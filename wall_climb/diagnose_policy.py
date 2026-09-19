"""不选择最佳检查点；独立检查命令响应、隐层饱和和执行裁剪。"""
import json
from pathlib import Path
import numpy as np
import torch
import config as C
from env import WallClimbEnv
from ppo import PPO


def diagnose(checkpoint, output):
    torch.set_num_threads(1)
    policy = PPO(C.OBS_DIM,C.ACT_DIM)
    policy.load(checkpoint)
    trials=[]
    for command in (-.5,0.,.5):
        env=WallClimbEnv(30000)
        env.set_conditions(90,.85)
        env.iter=policy.iteration
        obs=env.reset(); env.v_desired=command
        env.obs_filtered=env._raw_obs()
        clips=[]; saturated=[]; speeds=[]
        for _ in range(C.MAX_STEPS):
            obs=env.observe()
            action,_,_=policy.act(obs,deterministic=True,privileged=env.privileged())
            env.set_contact_estimate(policy.last_contact)
            with torch.no_grad():
                inputs=policy.actor_observation(torch.as_tensor(obs))[0]
                x=policy.actor.norm(inputs) if hasattr(policy.actor,'norm') else inputs
                layer_saturation=[]
                for layer in policy.actor.net:
                    x=layer(x)
                    if isinstance(layer,torch.nn.Tanh):
                        layer_saturation.append(float((x.abs()>.99).float().mean()))
                saturated.append(layer_saturation)
            clips.append(float(np.mean(np.abs(action)>1)))
            _,_,done,info=env.step(action); speeds.append(env.v_b)
            if done: break
        trials.append(dict(command=command,time=info['time'],mean_velocity=float(np.mean(speeds)),
                           rmse=info['velocity_rmse'],survived=info['survived'],
                           execution_clip_fraction=float(np.mean(clips)),
                           hidden_saturation_fraction=np.mean(saturated,axis=0).tolist()))
    result=dict(model_version=C.MODEL_VERSION,checkpoint=str(checkpoint),trials=trials)
    Path(output).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    import sys
    diagnose(sys.argv[1],sys.argv[2])
