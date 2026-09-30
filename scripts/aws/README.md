# Training on AWS (EC2)

Runs the full baseline training (`scripts/train_baseline_qg.py --max-train-samples -1`,
the whole official 87599-example SQuAD train split) on a `g4dn.xlarge` (NVIDIA T4,
16GB VRAM) instead of the local 4GB GTX 1650, where a full run is estimated at
17-28+ hours. On a T4 that's estimated at roughly 6-10 hours.

## Order of operations

```bash
cd scripts/aws
./01_launch_instance.sh       # launches the GPU instance (see cost note below)
./02_deploy_and_train.sh      # syncs the repo, sets up the venv, starts training
./03_check_status.sh          # re-run anytime to check progress
./04_fetch_results.sh         # pulls the checkpoint + logs back locally
./05_terminate_instance.sh    # stops billing -- run this as soon as you're done
```

Then locally:

```bash
python scripts/evaluate_qg.py --checkpoint outputs/checkpoints/baseline_qg_full_from_aws/final
```

## Before running `01_launch_instance.sh`

New/unused AWS accounts default to **0 vCPU quota** for GPU (G/VT family)
instances -- `01_launch_instance.sh` will refuse to launch until this is
raised. Check status with:

```bash
aws service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA --region us-east-1
aws service-quotas list-requested-service-quota-change-history --service-code ec2 --region us-east-1
```

If it's still 0 and nothing is pending, request an increase (free, does not
launch or cost anything by itself -- just opens a request with AWS, which can
take anywhere from minutes to a day or more to approve on a fresh account):

```bash
aws service-quotas request-service-quota-increase --service-code ec2 --quota-code L-DB2E81BA --desired-value 8 --region us-east-1
```

## Cost and safety

- `g4dn.xlarge` is on-demand-billed by the hour while running -- check current
  pricing before launching. A full run was estimated at a few dollars total.
- `01_launch_instance.sh` sets `--instance-initiated-shutdown-behavior=terminate`
  and schedules a **hard safety auto-shutdown** (default 12h, via user-data)
  so a forgotten instance can't run up an unbounded bill. If training
  legitimately needs more time, SSH in and run `sudo shutdown -c` before the
  deadline to cancel it.
- **`05_terminate_instance.sh` is what actually stops billing.** Don't rely on
  the safety shutdown -- run it yourself as soon as you have your results.
- State (`.instance_id`, `.instance_ip`) and the generated SSH key
  (`ic-tcqg-training-key.pem`) live in this directory and are gitignored --
  never commit the `.pem` file.
