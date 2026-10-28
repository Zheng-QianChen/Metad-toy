"""用真实 MPI communicator 验证同温度 umbrella 窗口交换。

world 覆盖全部 rank；replica 覆盖一个物理副本的 rank；leaders 仅含每个
副本的 rank 0。示例每个副本持有一个标量坐标，完整 MD 的状态可按相同
通信边界扩展为坐标、速度和积分器状态。MPI rank 是独立进程。
"""
import argparse
from pathlib import Path
import sys
import numpy as np
from mpi4py import MPI
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sampling.bias import HarmonicUmbrella,exchange_log_ratio
from common.checks import write_result


def run(output,ranks_per_replica):
    """运行 200 步模型采样与交替邻窗交换，写入结果 JSON。
    
    ranks_per_replica 为每副本进程数，必须整除 world.size，且至少得到两个
    副本。窗口中心随 replica_id 固定；walker 标签随被交换的配置移动。
    每副本 leader 更新其坐标，其他成员通过 replica.bcast 获取同一状态。
    断言分别检查副本内一致性、全局 walker 标签排列和接受/拒绝分支。
    """
    world=MPI.COMM_WORLD
    if world.size%ranks_per_replica or world.size//ranks_per_replica<2:
        raise ValueError('Need at least two equal-size replicas')
    # 连续的一组 world rank 属于同一副本；例如 4 rank / 2 得到两个副本。
    replica_id=world.rank//ranks_per_replica
    replica=world.Split(replica_id,world.rank)
    # MPI.UNDEFINED 让非 leader 得到空 communicator；只有 leader 调用其集合通信。
    leaders=world.Split(0 if replica.rank==0 else MPI.UNDEFINED,replica_id)
    count=world.size//ranks_per_replica
    centers=np.linspace(-1.,1.,count)
    rng=np.random.default_rng(1200+replica_id)
    coordinate=float(centers[replica_id]); walker=replica_id
    accepted=0; rejected=0; move_count=0
    for step in range(200):
        if replica.rank==0:
            # 相同的基础势、温度；各副本窗口固定，交换坐标和 walker 标签。
            def total(x): return .5*x*x+2*(x-centers[replica_id])**2
            candidate=coordinate+rng.normal(0.,.3)
            if np.log(rng.random())<total(coordinate)-total(candidate): coordinate=candidate
            # 保存交换前全部状态，使一对窗口的双方用同一份旧配置计算接受率。
            states=leaders.allgather((coordinate,walker))
            # 偶数轮配对 (0,1),(2,3)，奇数轮配对 (1,2),(3,4)，每轮窗口只出现一次。
            parity=step%2
            lower=replica_id if replica_id%2==parity else replica_id-1
            upper=lower+1
            if lower>=0 and upper<count:
                left=HarmonicUmbrella([centers[lower]],4.)
                right=HarmonicUmbrella([centers[upper]],4.)
                ratio=exchange_log_ratio(left,right,[states[lower][0]],[states[upper][0]],1.)
                # 每对使用同一随机数，以便双方做出一致的决定。
                uniform=np.random.default_rng(90000+step*count+lower).random()
                decision=np.log(uniform)<min(0.,ratio)
                if decision:
                    partner=upper if replica_id==lower else lower
                    coordinate,walker=states[partner]
                # 每对交换只由编号较小的 leader 计数，随后 world.reduce 可直接求和。
                if replica_id==lower:
                    accepted+=int(decision);rejected+=int(not decision)
            labels=leaders.allgather(walker)
            assert sorted(labels)==list(range(count)), 'Lost or duplicated replica configuration'
            move_count+=int(walker!=replica_id)
        # 交换决定完成后再在副本内广播；副本中的所有 rank 从相同状态开始下一步。
        coordinate,walker=replica.bcast((coordinate,walker),root=0)
        all_members=replica.allgather((coordinate,walker))
        assert all(state==all_members[0] for state in all_members)
    total_accepted=world.reduce(accepted,root=0);total_rejected=world.reduce(rejected,root=0)
    moves=world.reduce(move_count,root=0)
    if world.rank==0:
        assert total_accepted>0 and total_rejected>0 and moves>0
        write_result(output,f'replica_{count}x{ranks_per_replica}',{'world_ranks':world.size,'replicas':count,'ranks_per_replica':ranks_per_replica,'exchange_accepted':total_accepted,'exchange_rejected':total_rejected,'replica_consistency':True,'walker_permutation_preserved':True})
    if replica.rank==0: leaders.Free()
    replica.Free()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--ranks-per-replica',type=int,default=1)
    args=parser.parse_args()
    try: run(args.output,args.ranks_per_replica)
    # 任一 rank 验证失败时终止整个 world，避免其他 rank 卡在集合通信中。
    except Exception:
        import traceback;traceback.print_exc();MPI.COMM_WORLD.Abort(1)
