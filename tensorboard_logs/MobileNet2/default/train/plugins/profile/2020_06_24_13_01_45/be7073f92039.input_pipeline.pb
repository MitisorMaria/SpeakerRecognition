	�c\q�'p@�c\q�'p@!�c\q�'p@	�p���?�p���?!�p���?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�c\q�'p@xB�?���?A��c!�	p@Y-z�mC�?*	bX9�D�@2K
Iterator::Model::MapVdt@v�?!��|Z�W@)]��X3��?1É���V@:Preprocessing2X
!Iterator::Model::Map::ParallelMap$0��{�?!}��
�@)$0��{�?1}��
�@:Preprocessing2o
8Iterator::Model::Map::ParallelMap::Zip[1]::ForeverRepeat%��C��?!�!S\s�@)9�⪲�?1�Y��K@:Preprocessing2y
BIterator::Model::Map::ParallelMap::Zip[0]::FlatMap[0]::Concatenate�$�)� �?!�j�*�?)�N?���?1�{��2�?:Preprocessing2]
&Iterator::Model::Map::ParallelMap::Zip���Z(�?!�'U�:q@)[{��B�?1�)�p�?:Preprocessing2�
RIterator::Model::Map::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�ꫫ�x?!|�V&*��?)�ꫫ�x?1|�V&*��?:Preprocessing2F
Iterator::Model�7�Gn��?!���T�W@)�M�g\w?1E���ؑ�?:Preprocessing2{
DIterator::Model::Map::ParallelMap::Zip[1]::ForeverRepeat::FromTensor����i?!-ݘ_k��?)����i?1-ݘ_k��?:Preprocessing2i
2Iterator::Model::Map::ParallelMap::Zip[0]::FlatMap1�䠄�?!x�T� @)���;f?1����<{�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	xB�?���?xB�?���?!xB�?���?      ��!       "      ��!       *      ��!       2	��c!�	p@��c!�	p@!��c!�	p@:      ��!       B      ��!       J	-z�mC�?-z�mC�?!-z�mC�?R      ��!       Z	-z�mC�?-z�mC�?!-z�mC�?JCPU_ONLY