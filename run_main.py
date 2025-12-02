import subprocess
datasets = ['NASA', 'SUN']
models = ['UA_PINN', 'DeepAR']
for i, dataset in enumerate(datasets):
    for model in models:
        print(f"Running with dataset={dataset}...")
        print(f"Running with model={model}...")
        # 构造命令行参数
        command = [
            "python", './main.py',
            "--dataset", dataset,
            "--model", model,
            "--num_epochs", str(200),
            "--n_samples", str(200),
            "--train_batch_size", str(1024),
        ]
        print(" ".join(command))
        # 使用 subprocess 运行脚本
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        # 等待进程结束并获取返回码
        return_code = process.wait()
        print(f"Process finished with return code: {return_code}")
        print("-" * 40)


datasets = ['NASA', 'SUN']
models = ['CNN','MLP']
for i, dataset in enumerate(datasets):
    for model in models:
        print(f"Running with dataset={dataset}...")
        print(f"Running with model={model}...")
        # 构造命令行参数
        command = [
            "python", './main.py',
            "--dataset", dataset,
            "--model", model,
            "--num_epochs", str(200),
            "--n_samples", str(1),
            "--train_batch_size", str(256),
        ]
        print(" ".join(command))
        # 使用 subprocess 运行脚本
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        # 等待进程结束并获取返回码
        return_code = process.wait()
        print(f"Process finished with return code: {return_code}")
        print("-" * 40)
