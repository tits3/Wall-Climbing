"""模型修订的公式、可观测性和时间轴回归检查。"""
import math
import unittest
import numpy as np
import torch
from torch.distributions import Normal
import config as C
from env import WallClimbEnv
from ppo import PPO
from train import compute_gae
from rewards import paper_action, paper_reward
from kinematics import inverse, forward, nominal, action_for_pose

class CoreTests(unittest.TestCase):
    def test_airborne_leg_follows_body_without_false_joint_motion(self):
        e=self.vertical(); e.a[:]=0; e.x[:]=.08
        e.y=e.y_b+e.hip_offsets; e.joint_q=inverse(np.zeros(4),e.x)
        e.joint_velocity[:]=0; e.contact[:]=False; e.v_b=.5
        action=action_for_pose(np.zeros(4),e.x,np.zeros(4))
        before_q=e.joint_q.copy(); before_feet=e.y.copy(); before_body=e.y_b
        e._advance(action,C.DT)
        self.assertGreater(abs(e.y_b-before_body),.001)
        np.testing.assert_allclose(e.y-before_feet,e.y_b-before_body,atol=1e-9)
        np.testing.assert_allclose(e.joint_q,before_q,atol=1e-8)
        self.assertLess(np.linalg.norm(e.joint_acceleration),1e-4)
    def test_inverse_hip_branch_is_continuous(self):
        before=inverse(.001,.4)
        after=inverse(-.001,.4,reference=before)
        self.assertLess(np.max(np.abs(after-before)),.02)
        t,g=forward(after)
        np.testing.assert_allclose([t,g],[-.001,.4],atol=1e-12)
    def test_folded_inverse_retains_hip_reference(self):
        q=inverse(0.,C.BODY_NORMAL_HEIGHT,reference=np.array([.7,-math.pi]))
        self.assertAlmostEqual(q[0],.7)
        t,g=forward(q)
        np.testing.assert_allclose([t,g],[0.,C.BODY_NORMAL_HEIGHT],atol=1e-12)
    def test_legal_extension_does_not_switch_off_magnet(self):
        e=self.vertical()
        for _ in range(100):
            e.step(np.ones(C.ACT_DIM))
            self.assertTrue(e.a.all()); self.assertTrue(e.contact.all())
    def test_insufficient_friction_without_magnets_cannot_hold_slope(self):
        e=WallClimbEnv(); e.set_conditions(60,1,False); e.reset()
        action=action_for_pose(np.zeros(4),np.full(4,-.002),np.zeros(4))
        for _ in range(200):
            _,_,done,info=e.step(action)
            if done: break
        self.assertTrue(info['fell']); self.assertLess(info['time'],5)
    def test_no_modeling_matches_full_when_magnets_disabled(self):
        a=WallClimbEnv(123,domain_randomization=True)
        b=WallClimbEnv(123,domain_randomization=True,variant='no_modeling')
        rng=np.random.default_rng(9)
        for _ in range(50):
            action=rng.uniform(-1,1,C.ACT_DIM)
            oa,ra,da,_=a.step(action); ob,rb,db,_=b.step(action)
            np.testing.assert_array_equal(oa,ob); self.assertEqual(ra,rb); self.assertEqual(da,db)
    def vertical(self, p=1):
        e=WallClimbEnv(123); e.set_conditions(90,p); e.reset(); return e
    def command(self, magnet=1):
        return np.tile([0,0,magnet],4)
    def test_dimensions(self):
        e=self.vertical(); self.assertEqual(e.observe().shape,(C.OBS_DIM,)); self.assertEqual(C.ACT_DIM,12)
    def test_kinematics_roundtrip(self):
        tangent=np.array([-.1,0,.05,.1]); gap=np.array([0,.01,.04,.08])
        t,h=forward(inverse(tangent,gap)); np.testing.assert_allclose(t,tangent,atol=1e-14); np.testing.assert_allclose(h,gap,atol=1e-14)
    def test_joint_target_domain_covers_paper_commands_and_swing_height(self):
        for command in C.COMMAND_RANGE:
            tangent=np.full(4,command*C.GAIT_PERIOD*.375)
            for gap in (-C.CONTACT_GAP,C.SWING_HEIGHT):
                action=action_for_pose(tangent,np.full(4,gap),np.ones(4))
                from kinematics import joint_targets
                t,h=forward(joint_targets(action))
                np.testing.assert_allclose(t,tangent,atol=1e-7); np.testing.assert_allclose(h,gap,atol=1e-7)
    def test_running_normalization_roundtrip_save_load(self):
        from ppo import RunningNorm
        n=RunningNorm(2); n.update(torch.tensor([[1.,2.],[3.,4.]]))
        self.assertLess(float(n(torch.tensor([2.,3.])).abs().max()),.001)
        clone=RunningNorm(2); clone.load_state_dict(n.state_dict())
        torch.testing.assert_close(n(torch.tensor([5.,6.])),clone(torch.tensor([5.,6.])))
    def test_action_rescaling_preserves_initial_physical_exploration(self):
        p=PPO(C.OBS_DIM,C.ACT_DIM)
        _,std=p.actor(torch.zeros(C.OBS_DIM))
        joint=[i for i in range(C.ACT_DIM) if i%3!=2]
        np.testing.assert_allclose(std.detach().numpy()[joint]*C.JOINT_ACTION_SCALE,.5*math.exp(-1),rtol=1e-6)
    def test_mechanical_stance_foot_retains_world_anchor(self):
        e=WallClimbEnv(); old=e.y.copy()
        action=action_for_pose(np.full(4,-.03),np.full(4,-.002),np.zeros(4))
        e.step(action)
        np.testing.assert_array_equal(e.y,old)
        self.assertTrue(e.contact.all()); self.assertEqual(e.a.sum(),0)
    def test_air_gap_is_not_mechanical_support(self):
        e=WallClimbEnv(); e.x[:]=.001; e.joint_q=inverse(np.zeros(4),e.x)
        n=e._advance(self.command(-1),0)
        self.assertEqual(n,0)
    def test_failure_sources_are_separated(self):
        e=self.vertical(); e.disturb(0); self.assertEqual(e.pending_sources[0],'forced_slip')
        e._failure(1,'attachment failed'); self.assertEqual(e.pending_sources[1],'probabilistic')
    def test_forced_event_timestamp_is_current_simulation_time(self):
        e=self.vertical(); e.step_count=150; e.event_time=2.987
        e.disturb(0); self.assertEqual(e.pending[0],3.)
    def test_ideal_adhesion_does_not_stretch_mechanical_links(self):
        e=self.vertical(); e.variant='no_modeling'
        for _ in range(100):
            e.step(np.ones(C.ACT_DIM))
            relative=e.y-e.y_b-e.hip_offsets
            radius=np.sqrt(relative**2+(C.BODY_NORMAL_HEIGHT-e.x)**2)
            self.assertLessEqual(float(radius.max()),sum(C.LINK_LENGTHS)+1e-9)
            t,h=forward(e.joint_q)
            np.testing.assert_allclose(t,relative,atol=1e-8); np.testing.assert_allclose(h,e.x,atol=1e-8)
    def test_bulk_critic_inference_matches_per_step(self):
        p=PPO(C.OBS_DIM,C.ACT_DIM)
        obs=np.random.default_rng(9).normal(size=(64,C.OBS_DIM)).astype(np.float32)
        batch=p.value_batch(obs)
        single=np.array([p.value(row) for row in obs])
        np.testing.assert_allclose(batch,single,atol=2e-6,rtol=2e-6)
    def test_mechanical_swing_can_leave_world_anchor(self):
        e=WallClimbEnv(); old=e.y.copy()
        a=action_for_pose(np.full(4,.1),np.full(4,.08),np.zeros(4))
        for _ in range(30): e.step(a)
        self.assertTrue(np.any(e.x>C.CONTACT_GAP)); self.assertTrue(np.any(np.abs(e.y-old)>1e-4))
    def test_one_mm_is_not_alignment(self):
        e=self.vertical(); e.a[:]=0; e.x[:]=.001; e.joint_q=inverse(np.zeros(4),e.x)
        e._advance(self.command(),0); self.assertEqual(e.a.sum(),0)
    def test_zero_gap_allows_alignment(self):
        e=self.vertical(); e.a[:]=0; e._advance(self.command(),0); self.assertEqual(e.a.sum(),4)
    def test_no_modeling_ignores_geometry(self):
        e=self.vertical(); e.variant='no_modeling'; e.a[:]=0; e.x[:]=.04; e.joint_q=inverse(np.zeros(4),e.x); e.y+=2*C.LEG_REACH
        e._advance(self.command(),0); self.assertEqual(e.a.sum(),4); np.testing.assert_allclose(e.x,.04)
        self.assertFalse(e.contact.any())
    def test_magnet_does_not_command_foot_height(self):
        a,b=self.vertical(),self.vertical()
        for e in (a,b):
            e.a[:]=0; e.x[:]=.04; e.joint_q=inverse(np.zeros(4),e.x)
        a._advance(self.command(1),C.DT); b._advance(self.command(-1),C.DT)
        np.testing.assert_allclose(a.x,b.x)
    def test_actor_has_no_internal_attachment_retry_information(self):
        e=self.vertical(); a=e._raw_obs(); e.a[:]=0; e.retry[:]=100; e.failed_activation[:]=True
        np.testing.assert_array_equal(a,e._raw_obs())
    def test_rotated_gravity_is_not_proprioceptive_orientation(self):
        e=self.vertical(); before=e._raw_obs(); e.set_conditions(45); after=e._raw_obs()
        np.testing.assert_array_equal(before,after)
        self.assertAlmostEqual(e.privileged()[2],math.sqrt(.5),places=6)
    def test_estimator_cannot_read_truth(self):
        p=PPO(C.OBS_DIM,C.ACT_DIM); a=torch.zeros(C.OBS_DIM); b=a.clone(); b[0]=100
        b[C.CONTACT_OBS_INDICES]=1; b[C.HEIGHT_OBS_INDICES]=1
        torch.testing.assert_close(p.estimator(a),p.estimator(b)); torch.testing.assert_close(p.actor_observation(a)[0],p.actor_observation(b)[0])
    def test_contact_estimator_gate(self):
        e=self.vertical(); e.a[0]=0; e.set_contact_estimate([.49,1,1,1]); e._advance(self.command(),0); self.assertFalse(e.a[0])
        e.set_contact_estimate([.51,1,1,1]); e._advance(self.command(),0); self.assertTrue(e.a[0])
    def test_probability(self):
        e=self.vertical(.85); n=0
        for _ in range(1000):
            e.reset(); e.a[0]=0; e._advance(self.command(),0); n+=e.a[0]
        self.assertLess(abs(n/1000-.85),.04)
    def test_no_automatic_resampling_after_failed_activation(self):
        e=self.vertical(0); e.a[0]=0; e._advance(self.command(),0); n=e.attempts
        for _ in range(10): e._advance(self.command(),0)
        self.assertEqual(e.attempts,n)
        e._advance(self.command(-1),0); e._advance(self.command(),0); self.assertGreater(e.attempts,n)
    def test_probabilistic_failure_is_not_geometric_detachment(self):
        e=self.vertical(0); e.a[0]=0; e._advance(self.command(),0)
        self.assertEqual(e.x[0],0); self.assertTrue(e.contact[0]); self.assertFalse(e.a[0])
    def test_recovery_requires_survival(self):
        e=self.vertical(); e.recovery_delays=[.2]; e.episode_terminated=True
        self.assertEqual(e.metrics()['recovery']['1.2'],0); self.assertEqual(e.metrics()['reattachment']['1.2'],1)
    def test_forced_slip_preserves_filter_history(self):
        e=self.vertical(); before=e.observe(); e.disturb(0); np.testing.assert_array_equal(before,e.observe())
    def test_observation_history_contains_joint_targets_only(self):
        e=self.vertical(); before=e._raw_obs()
        e.prev_action[2::3]=1; e.older_action[2::3]=-1
        np.testing.assert_array_equal(before,e._raw_obs())
        e.prev_action[0]=.25; e.older_action[1]=-.25
        after=e._raw_obs()
        self.assertEqual(after.shape,(58,))
        np.testing.assert_allclose(after[42:50],paper_action(e.prev_action).reshape(4,3)[:,:2].ravel())
        np.testing.assert_allclose(after[50:58],paper_action(e.older_action).reshape(4,3)[:,:2].ravel())
        self.assertNotEqual(float(before[42]),float(after[42]))
    def test_physical_action_units(self):
        a=self.command(-1); b=self.command(1); self.assertAlmostEqual(np.square(paper_action(a)-paper_action(b)).sum(),4)
        q=paper_action(self.command()).reshape(4,3)[:,:2]; np.testing.assert_allclose(q,nominal())
    def test_reward_hand_calculated(self):
        phase=np.array([.2,2,3,5]); contact=np.array([0,1,1,1],bool); z=np.zeros(C.ACT_DIM)
        r,t=paper_reward(0,phase,contact,np.array([.08,0,0,0]),np.zeros(4),np.zeros(4),.12,.12,z,z,z,contact.astype(float))
        self.assertAlmostEqual(r,6.5*math.exp(-.6)); self.assertEqual(t['Ras1'],0)
    def test_joint_reward_units_and_weights(self):
        z=np.zeros(C.ACT_DIM); q=nominal()+.1
        _,t=paper_reward(0,np.array([.2,2,3,5]),np.ones(4,bool),np.zeros(4),np.zeros(4),np.zeros(4),.12,.12,z,z,z,np.ones(4),joint_position=q,joint_velocity=np.ones((4,2)),joint_acceleration=2*np.ones((4,2)),torque=3*np.ones((4,2)))
        self.assertAlmostEqual(t['Rjp'],.75*8*.01)
        self.assertAlmostEqual(t['Rjs'],.003*8)
        self.assertAlmostEqual(t['Rja'],.003*8*4)
        self.assertAlmostEqual(t['Rtau'],.003*8*9)
    def test_curriculum(self):
        e=self.vertical(); e.set_curriculum(1200); self.assertFalse(e.adhesion_enabled)
        e.set_curriculum(35000); self.assertAlmostEqual(e.theta,math.pi/2); self.assertAlmostEqual(e.p_attach,.85)
    def test_stuck_overrides_timeout(self):
        e=self.vertical(); e.step_count=C.MAX_STEPS-1; e.stuck_steps=C.STUCK_STEPS
        _,_,done,i=e.step(self.command()); self.assertTrue(done); self.assertTrue(i['stuck']); self.assertFalse(i['survived'])
    def test_delay_execution_order(self):
        e=self.vertical(); e.action_delay=.008; seen=[]; original=e._advance
        def spy(a,dt): seen.append((a.copy(),dt)); return original(a,dt)
        e._advance=spy; a=self.command(-1); old=e.prev_action.copy(); e.step(a)
        np.testing.assert_array_equal(seen[0][0],old); np.testing.assert_array_equal(seen[1][0],a)
        self.assertAlmostEqual(seen[0][1],.008); self.assertAlmostEqual(seen[1][1],.012)
    def test_timeout_gae(self):
        _,r=compute_gae(np.array([1.,2.]),np.array([.5,.7]),np.array([1.,0.]),0,next_values=np.array([.9,.8]),terminated=np.zeros(2))
        self.assertAlmostEqual(float(r[0]),1+C.GAMMA*.9,places=5)
    def test_independent_batched_gae(self):
        r=np.array([[1.,100.],[2.,200.]]); d=np.array([[1.,0.],[0.,1.]]); v=np.zeros_like(r)
        b,_=compute_gae(r,v,d,np.zeros(2))
        for i in range(2):
            a,_=compute_gae(r[:,i],v[:,i],d[:,i],0); np.testing.assert_allclose(b[:,i],a)
    def test_raw_gaussian_log_probability(self):
        p=PPO(C.OBS_DIM,C.ACT_DIM); o=self.vertical().observe(); a,lp,_=p.act(o)
        mu,s=p.actor(p.actor_observation(torch.as_tensor(o))[0]); expected=Normal(mu,s).log_prob(torch.as_tensor(a)).sum()
        self.assertAlmostEqual(lp,expected.item(),places=5)
    def test_seed_repeatability(self):
        a,b=self.vertical(.85),self.vertical(.85)
        for _ in range(20):
            oa,ra,da,_=a.step(self.command()); ob,rb,db,_=b.step(self.command())
            np.testing.assert_array_equal(oa,ob); self.assertEqual((ra,da),(rb,db))

if __name__=='__main__':
    torch.set_num_threads(1); unittest.main()
