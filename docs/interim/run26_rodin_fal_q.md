# Run #26 中间数据存档(低模轮第一批,2026-07-20)

rodin_fal_q(Rodin 2.5 quad 模式,7/8 完成,1 超时):
- 生产分 92.4(CI 91-93)| 游戏榜 83.1(预算 30.0)| 水密焊接 7/7
- 直绑 6/7,+remesh 7/7,烟测 100 | 切片 manifold 7/7 | **VLM 语义 1.0(全场最低)**
- 分类:character 92.7 / hardsurface 91.0 / organic 93.9 / prop 87.7 | 时延 172s

同轮失败(已修复,待重跑):
- meshy_lp 8×400:smart-topology 需 ai_model meshy-t1/t2(已改 t2;提交失败未扣费)
- tripo_h31q 8×下载失败:quad 输出为 FBX(已加 Kaydara 魔数识别 + bpy 自动转 GLB;
  8 次生成已扣费 ~300cr,FBX 文件存于 #26 artifact)
