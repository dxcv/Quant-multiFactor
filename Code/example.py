import conf_func

func=conf_func.func_caculate_factor

if __name__=='__main__':
    #单因子测试
    #func['ic_test']('size', 20080102, 20121231)
    #func['backtest']('size', 20090102, 20091231)

    #大类因子合成
    func['fct_weight']('value', 20080102, 20121231)

    #大类因子测试
    #func['ic_test']('value',20091102,20091231,fct_class='class',weight_method='weight_IC')
    #func['backtest']('value',20091102,20091231,fct_class='class',weight_method='weight_IC')
