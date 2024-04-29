import json

if __name__ == '__main__':

    ll=['a','b','c','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']
    print('xxx'.join(ll))
    temp = '{"fix_desc": "经检查，室外机电控盒坏。更换外机电控盒。加装电抗器后测试正常", "fix_name": "开机有显示，但不制冷/制热，室外电控板坏，更换", "err_kind": "电器电控类", "err_reason": "室外电控板"}{"fix_desc": "电抗器噪音大，更换", "fix_name": "压缩机噪音大，更换", "err_kind": "制冷系统类", "err_reason": "压缩机"}{"fix_desc": "保修电子室内外主板升级，旧件回收。", "fix_name": "更换室内主板，旧件回收。", "err_kind": "电器电控类", "err_reason": "室内电控板"}{"fix_desc": "更换室外机电控一个旧件上交罗家园子网点", "fix_name": "开机有显示，但不制冷/制热，室外电控板坏，更换", "err_kind": "电器电控类", "err_reason": "室外电控板"}{"fix_desc": "更换毛细管后成", "fix_name": "主毛细管阻值大（长），更换", "err_kind": "制冷系统类", "err_reason": "毛细管"}'

    split = temp.split('}{')
    result = ''
    for index, s in enumerate(split):
        if index == 0:
            s += '}'
        elif index == len(split) - 1:
            s = '{' + s
        else:
            s = '{' + s + '}'
        try:
            loads = json.loads(s)
            print(loads)
            result += loads['fix_name']+"\n"
        except Exception:
            print(s)
            print('error')

    print(result)
