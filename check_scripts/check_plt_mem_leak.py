import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
from multiprocessing import Process
import gc
import objgraph
from memory_profiler import profile

def plt_func():
    save_path = 'tmp/tmp.jpg'
    fig, ax = plt.subplots(1,2, figsize=(14,6), dpi=200)
    ax[0].axis('off')
    ax[0].set_title(f'DoLP')
    ax_0 = ax[0].imshow(np.random.rand(512,512), cmap='GnBu', vmax=1, vmin=0)
    bar_0 = fig.colorbar(ax_0, ax=ax[0])
    ax[1].axis('off')
    ax[1].set_title(f'AoLP')
    ax_1 = ax[1].imshow(np.random.rand(512,512), cmap='GnBu', vmax=1, vmin=0)
    bar_1 = fig.colorbar(ax_1, ax=ax[1])
    fig.tight_layout()
    fig.savefig(save_path)
    fig.clf()
    plt.close()
    gc.collect()

def iter_plt_func():
    objgraph.show_growth()
    p = Process(target=plt_func)
    p.start()
    p.join()
    print('iter after')
    objgraph.show_growth()
    p = Process(target=plt_func)
    p.start()
    p.join()
    print('iter after')
    objgraph.show_growth()
    p = Process(target=plt_func)
    p.start()
    p.join()
    print('iter after')
    objgraph.show_growth()
    p = Process(target=plt_func)
    p.start()
    p.join()
    print('iter after')
    objgraph.show_growth()
    # roots = objgraph.get_leaking_objects()
    # objgraph.show_most_common_types(objects=roots)
    # print()
    # plt_func()
    # print('after iter')
    # objgraph.show_growth()
    # roots = objgraph.get_leaking_objects()
    # objgraph.show_most_common_types(objects=roots)
    # objgraph.show_refs(roots[:4], refcounts=True, filename='roots.png')
    
if __name__ == '__main__':
    gc.collect()
    iter_plt_func()
        # print()