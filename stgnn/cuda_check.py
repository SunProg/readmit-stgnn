import os

import dgl
import torch


def main():
    print("CUDA preflight")
    print("CUDA_VISIBLE_DEVICES={}".format(os.environ.get("CUDA_VISIBLE_DEVICES", "unset")))
    print("torch.__version__={}".format(torch.__version__))
    print("torch.version.cuda={}".format(torch.version.cuda))
    print("torch.cuda.is_available()={}".format(torch.cuda.is_available()))
    print("torch.cuda.device_count()={}".format(torch.cuda.device_count()))

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot see CUDA.")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    print("device_name={}".format(torch.cuda.get_device_name(device)))

    x = torch.randn((2048, 2048), device=device)
    y = torch.randn((2048, 2048), device=device)
    z = x @ y
    torch.cuda.synchronize(device)
    print("matmul_sum={:.6f}".format(z.sum().item()))
    print(
        "torch_cuda_memory_mib allocated={:.1f} reserved={:.1f}".format(
            torch.cuda.memory_allocated(device) / 1024 / 1024,
            torch.cuda.memory_reserved(device) / 1024 / 1024,
        )
    )

    graph = dgl.graph(([0, 1], [1, 2]), num_nodes=3).to(device)
    print("dgl_graph_device={}".format(graph.device))
    print("CUDA preflight OK")


if __name__ == "__main__":
    main()
