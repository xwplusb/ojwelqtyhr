# ojwelqtyhr
- [x] teacher net and trainer
- [x] score net and trainer
- [x] student's selective targets training
- [o] something called genetic algorithm optimizing :worried:
- [ ] visualize the results :fearful:
- [ ] tune all the nets and parameters :skull:
- [ ] debug and tests


## 传输模型与传输图片对比实验设计
实验需要体现出图片传输的优越性
#### 数据量对比
当前模型大小只有9.5MB，直接传输模型肯定比传输图片效率高得多，200多张图片约50MB。但是可以对目标类只传输一张图片，学生模型过拟合依然可以通过教师模型的检验
#### 多样性对比
