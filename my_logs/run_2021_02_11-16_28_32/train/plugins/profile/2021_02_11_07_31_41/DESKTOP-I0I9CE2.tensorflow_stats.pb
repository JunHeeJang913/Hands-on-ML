"�&
uHostFlushSummaryWriter"FlushSummaryWriter(13333s��@93333s��@A3333s��@I3333s��@a��xN�9�?i��xN�9�?�Unknown�
BHostIDLE"IDLE1fffff��@Afffff��@a�V�T�?i�N�P��?�Unknown
tHost_FusedMatMul"sequential_5/dense_28/Relu(1ffffff.@9ffffff.@Affffff.@Iffffff.@a��h)L?iI��cQ��?�Unknown
iHostWriteSummary"WriteSummary(1      +@9      +@A      +@I      +@aK���H?i�k�����?�Unknown�
�HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1ffffff.@9ffffff.@A333333)@I333333)@a����8G?i���W��?�Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1������!@9������!@A������!@I������!@a�G{��7@?i�6�e��?�Unknown
�HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1ffffff)@9ffffff)@A������ @I������ @a��ک�>?i��=�8��?�Unknown
~HostMatMul"*gradient_tape/sequential_5/dense_29/MatMul(1������@9������@A������@I������@aL���a<?i�n�����?�Unknown
�	HostMatMul",gradient_tape/sequential_5/dense_29/MatMul_1(1������@9������@A������@I������@a�U�lrF;?iy(�-��?�Unknown
~
HostMatMul"*gradient_tape/sequential_5/dense_30/MatMul(1������@9������@A������@I������@a�U�lrF;?id�u}���?�Unknown
eHost
LogicalAnd"
LogicalAnd(1������@9������@A������@I������@a���p��:?i������?�Unknown�
~HostMatMul"*gradient_tape/sequential_5/dense_28/MatMul(1������@9������@A������@I������@a���p��:?i��Q�8��?�Unknown
dHostDataset"Iterator::Model(1ffffff/@9ffffff/@A������@I������@a1S���n9?i��p�f��?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1������@9������@A������@I������@a1S���n9?i&������?�Unknown
^HostGatherV2"GatherV2(1������@9������@A������@I������@aX���(�6?i���o��?�Unknown
�HostDynamicStitch".gradient_tape/mean_squared_error/DynamicStitch(1������@9������@A������@I������@aX���(�6?i��� K��?�Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a���E4?i������?�Unknown
�HostTile"5gradient_tape/mean_squared_error/weighted_loss/Tile_1(1ffffff@9ffffff@Affffff@Iffffff@aq�:�2?i�U�\-��?�Unknown
gHostStridedSlice"strided_slice(1      @9      @A      @I      @a.���m2?i7��{��?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1������@9������@A������@I������@a�I��2?i`�-	���?�Unknown
�HostSquaredDifference"$mean_squared_error/SquaredDifference(1333333@9333333@A333333@I333333@a�|��'�1?i0�!.���?�Unknown
�HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1������@9������@A������@I������@a�G{��70?i��V%���?�Unknown
tHost_FusedMatMul"sequential_5/dense_29/Relu(1������@9������@A������@I������@a%Z�X�.?iOg�����?�Unknown
lHostIteratorGetNext"IteratorGetNext(1ffffff@9ffffff@Affffff@Iffffff@a���\N9.?i�5����?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a%�`�|-?i�C���?�Unknown
wHost_FusedMatMul"sequential_5/dense_30/BiasAdd(1333333@9333333@A333333@I333333@a���d�,?iV��p��?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1������@9������@A������@I������@a�U�lrF+?iK^5I%��?�Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1ffffff@@9ffffff@@Affffff
@Iffffff
@af�|�S(?i�)�����?�Unknown
�HostBiasAddGrad"7gradient_tape/sequential_5/dense_29/BiasAdd/BiasAddGrad(1������@9������@A������@I������@aX���(�&?i�t%%��?�Unknown
�HostReluGrad",gradient_tape/sequential_5/dense_29/ReluGrad(1������@9������@A������@I������@aX���(�&?i���ǅ��?�Unknown
�HostBiasAddGrad"7gradient_tape/sequential_5/dense_28/BiasAdd/BiasAddGrad(1333333@9333333@A333333@I333333@aJ����`%?i��V����?�Unknown
� HostBiasAddGrad"7gradient_tape/sequential_5/dense_30/BiasAdd/BiasAddGrad(1ffffff@9ffffff@Affffff@Iffffff@a�朐�$?i���&��?�Unknown
u!HostSum"$mean_squared_error/weighted_loss/Sum(1������@9������@A������@I������@a<L��L�#?i��X�d��?�Unknown
�"HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1������@9�������?A������@I�������?a�����*#?if�1���?�Unknown
�#HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1������@9������@A������@I������@a�����*#?i1�����?�Unknown
�$HostReluGrad",gradient_tape/sequential_5/dense_28/ReluGrad(1333333@9333333@A333333@I333333@a�|��'�!?i�������?�Unknown
�%HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1������,@9������,@A�������?I�������?a�P��ߖ?i4������?�Unknown
a&HostIdentity"Identity(1�������?9�������?A�������?I�������?a�P��ߖ?i     �?�Unknown�*�%
uHostFlushSummaryWriter"FlushSummaryWriter(13333s��@93333s��@A3333s��@I3333s��@a��ۅӜ�?i��ۅӜ�?�Unknown�
tHost_FusedMatMul"sequential_5/dense_28/Relu(1ffffff.@9ffffff.@Affffff.@Iffffff.@a�[¦�LN?i0[E�f��?�Unknown
iHostWriteSummary"WriteSummary(1      +@9      +@A      +@I      +@a�2D�U�J?i=,�!��?�Unknown�
�HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1ffffff.@9ffffff.@A333333)@I333333)@a��.�I?i�w��h��?�Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1������!@9������!@A������!@I������!@a�}{ӊA?ibW�K˵�?�Unknown
�HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1ffffff)@9ffffff)@A������ @I������ @aާ`��@?i�^.6��?�Unknown
~HostMatMul"*gradient_tape/sequential_5/dense_29/MatMul(1������@9������@A������@I������@ag~�>?ibq�Ľ�?�Unknown
�HostMatMul",gradient_tape/sequential_5/dense_29/MatMul_1(1������@9������@A������@I������@a����=?i�㉬t��?�Unknown
~	HostMatMul"*gradient_tape/sequential_5/dense_30/MatMul(1������@9������@A������@I������@a����=?i�e��$��?�Unknown
e
Host
LogicalAnd"
LogicalAnd(1������@9������@A������@I������@a��Y៴<?iܐ�X���?�Unknown�
~HostMatMul"*gradient_tape/sequential_5/dense_28/MatMul(1������@9������@A������@I������@a��Y៴<?i���Q��?�Unknown
dHostDataset"Iterator::Model(1ffffff/@9ffffff/@A������@I������@aLgK�n�;?i�el:���?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1������@9������@A������@I������@aLgK�n�;?i�>�2��?�Unknown
^HostGatherV2"GatherV2(1������@9������@A������@I������@as�s��8?i~���I��?�Unknown
�HostDynamicStitch".gradient_tape/mean_squared_error/DynamicStitch(1������@9������@A������@I������@as�s��8?i�`��?�Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a�~]Z��5?i�O&8��?�Unknown
�HostTile"5gradient_tape/mean_squared_error/weighted_loss/Tile_1(1ffffff@9ffffff@Affffff@Iffffff@aM���FU4?iZ�����?�Unknown
gHostStridedSlice"strided_slice(1      @9      @A      @I      @ajК#6�3?i�a��&��?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1������@9������@A������@I������@a��@�%�3?iʩs���?�Unknown
�HostSquaredDifference"$mean_squared_error/SquaredDifference(1333333@9333333@A333333@I333333@a���@#3?i��O���?�Unknown
�HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1������@9������@A������@I������@a�}{ӊ1?i[6��-��?�Unknown
tHost_FusedMatMul"sequential_5/dense_29/Relu(1������@9������@A������@I������@aO�ɘ��0?i�O�E��?�Unknown
lHostIteratorGetNext"IteratorGetNext(1ffffff@9ffffff@Affffff@Iffffff@ak�o'�X0?i�=#�P��?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a�*l#�/?i3 Z�N��?�Unknown
wHost_FusedMatMul"sequential_5/dense_30/BiasAdd(1333333@9333333@A333333@I333333@aJ�v�/?i���v@��?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1������@9������@A������@I������@a����-?i~؎���?�Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1ffffff@@9ffffff@@Affffff
@Iffffff
@a��<9=P*?iNlb����?�Unknown
�HostBiasAddGrad"7gradient_tape/sequential_5/dense_29/BiasAdd/BiasAddGrad(1������@9������@A������@I������@as�s��(?i��I��?�Unknown
�HostReluGrad",gradient_tape/sequential_5/dense_29/ReluGrad(1������@9������@A������@I������@as�s��(?i��Ѕ���?�Unknown
�HostBiasAddGrad"7gradient_tape/sequential_5/dense_28/BiasAdd/BiasAddGrad(1333333@9333333@A333333@I333333@a��k��'?i��k�F��?�Unknown
�HostBiasAddGrad"7gradient_tape/sequential_5/dense_30/BiasAdd/BiasAddGrad(1ffffff@9ffffff@Affffff@Iffffff@a���˘S&?i������?�Unknown
u HostSum"$mean_squared_error/weighted_loss/Sum(1������@9������@A������@I������@a�[�w�%?iJw2��?�Unknown
�!HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1������@9�������?A������@I�������?a1OW�$?i;~��O��?�Unknown
�"HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1������@9������@A������@I������@a1OW�$?i,�W����?�Unknown
�#HostReluGrad",gradient_tape/sequential_5/dense_28/ReluGrad(1333333@9333333@A333333@I333333@a���@##?i������?�Unknown
�$HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1������,@9������,@A�������?I�������?aڸ�V�?iۥ����?�Unknown
a%HostIdentity"Identity(1�������?9�������?A�������?I�������?aڸ�V�	?i�������?�Unknown�2CPU