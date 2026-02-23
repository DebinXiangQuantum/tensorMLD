 cudaqx/libs/qec/python/cudaq_qec/plugins/decoders/tensor_network_decoder.py 给出了用 tensor network 实现exact MLD的方式。
现在你的任务是理解张量网络解码实现MLD的原理，然后由于目前只是基于depolarizing error
  model，需要把ensor_network_decoder.py  扩展到 erasure error,注意仍然是circuit level model， 目前 erasureError/erasureerror.md 给出了可能的解决方案，你需要写代码验证该方案的正确性。
  请复制tensor_network_decoder.py 的相关解码代码到 erasureError/ 后，在其基础上增加对 erasure error 的处理，并测试其在不同码和不同erasure
  error情况的逻辑错误率。 有任何不清楚的细节请中断并询问我。最后希望采用python实现，整个实现代码保存在erasureError/目录下，不要更改其他目录的内容，其他目录是只读的，保持代码文件结构组织良好，源码和测试代码分离，中间输出必要的信息用于调试，同时要有合理的注释。 请制定计划并开始整个任务。
  
  为了验证该方法的完备性，需要验证两件事，1。纠错是有效果的，也就是随着物理错误率下降，逻辑错误率也下降，最后低于物理错误率，也就是达到纠错阈值之下，因此需
  要扫描不同物理错误率的结果，以及不同erasure错误分布的结果。2。为了证明MLD在擦除错误下仍然是最优的，还是需要跑其他的解码器,包括BP+OSD，Sequential Relay BP ，Min-Sum BP (bp_method=1)与目前实现的tensor
  network解码器进行对比，可以参考cudaqx/docs/sphinx/examples_rst/qec/decoders.rst 文档中的说明。需要注意的是解码器要在同一个错误模型设置下。
  所有结果以json形式存放到erasureError/results 下面，最后erasureError/plots下面新建绘图脚本绘制出 不同解码器在不同码下面，逻辑错误率（纵轴）随
  物理错误率的变化曲线。请仔细思考并拟定计划。
  不需要你重新实现，请直接调用cudaqx 的接口，让他们适应erasure错误模型就行。
 我看目前的结果是ler大于物理错误率这是什么原因，这已经意味着MLD解码失效了。 我们是可以知道erasure
  error发生的具体位置的，只是无法知道是否发生了什么类型的错误，同时我想知道erasure
  error是如何影响syndrome的，请你仔细思考其中的数学细节，调试检查目前的实现代码，解决这个问题。