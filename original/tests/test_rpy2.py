from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

def worker(x):
    from rpy2.robjects import r
    return r("sum(1:10)")[0]

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    with ProcessPoolExecutor() as ex:
        print(list(ex.map(worker, range(4))))
