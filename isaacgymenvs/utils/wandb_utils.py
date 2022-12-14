import wandb


def delete_last_run(run_id):
    api = wandb.Api()
    if run_id == "last":
        run = api.runs(path="krshna/isaacgymenvs", per_page=1)[0]
    else:
        run = api.run(path=f"krshna/isaacgymenvs/{run_id}")
    try:
        run.delete()
    except Exception:
        return False
    else:
        print(f"deleted run: {run.id}")
        return True
