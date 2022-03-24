from importance.fanova import HyperImpt

def main():
    pool = mp.Pool(6)
    start = timer()
    # responses = [pool.apply(train_xgb, args=(param, )) for param in rounded_list]
    for param in rounded_list:
        pool.apply_async(train_xgb, args=(param,), callback=collect_results)
        # responses.append(train_xgb(param))

    pool.close()
    pool.join()
    print(timer() - start)

if __name__ == '__main__':
    main()