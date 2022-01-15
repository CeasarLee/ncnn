// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "interp_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

Interp_vulkan::Interp_vulkan()
{
    support_vulkan = true;
    support_image_storage = true;

    pipeline_interp = 0;
    pipeline_interp_pack4 = 0;
    pipeline_interp_pack8 = 0;

    pipeline_interp_bicubic_coeffs_x = 0;
    pipeline_interp_bicubic_coeffs_y = 0;
    pipeline_interp_bicubic = 0;
    pipeline_interp_bicubic_pack4 = 0;
    pipeline_interp_bicubic_pack8 = 0;
}

int Interp_vulkan::create_pipeline(const Option& _opt)
{
    Option opt = _opt;
    const Mat& shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];
    const Mat& out_shape = top_shapes.empty() ? Mat() : top_shapes[0];

    int elempack = 1;
    if (shape.dims == 1) elempack = opt.use_shader_pack8 && shape.w % 8 == 0 ? 8 : shape.w % 4 == 0 ? 4 : 1;
    if (shape.dims == 2) elempack = opt.use_shader_pack8 && shape.h % 8 == 0 ? 8 : shape.h % 4 == 0 ? 4 : 1;
    if (shape.dims == 3) elempack = opt.use_shader_pack8 && shape.c % 8 == 0 ? 8 : shape.c % 4 == 0 ? 4 : 1;

    int out_elempack = 1;
    if (out_shape.dims == 1) out_elempack = opt.use_shader_pack8 && out_shape.w % 8 == 0 ? 8 : out_shape.w % 4 == 0 ? 4 : 1;
    if (out_shape.dims == 2) out_elempack = opt.use_shader_pack8 && out_shape.h % 8 == 0 ? 8 : out_shape.h % 4 == 0 ? 4 : 1;
    if (out_shape.dims == 3) out_elempack = opt.use_shader_pack8 && out_shape.c % 8 == 0 ? 8 : out_shape.c % 4 == 0 ? 4 : 1;

    size_t elemsize;
    size_t out_elemsize;
    if (opt.use_fp16_storage)
    {
        elemsize = elempack * 2u;
        out_elemsize = out_elempack * 2u;
    }
    else if (opt.use_fp16_packed)
    {
        elemsize = elempack == 1 ? 4u : elempack * 2u;
        out_elemsize = out_elempack == 1 ? 4u : out_elempack * 2u;
    }
    else
    {
        elemsize = elempack * 4u;
        out_elemsize = out_elempack * 4u;
    }

    Mat shape_packed;
    if (shape.dims == 1) shape_packed = Mat(shape.w / elempack, (void*)0, elemsize, elempack);
    if (shape.dims == 2) shape_packed = Mat(shape.w, shape.h / elempack, (void*)0, elemsize, elempack);
    if (shape.dims == 3) shape_packed = Mat(shape.w, shape.h, shape.c / elempack, (void*)0, elemsize, elempack);

    Mat out_shape_packed;
    if (out_shape.dims == 1) out_shape_packed = Mat(out_shape.w / out_elempack, (void*)0, out_elemsize, out_elempack);
    if (out_shape.dims == 2) out_shape_packed = Mat(out_shape.w, out_shape.h / out_elempack, (void*)0, out_elemsize, out_elempack);
    if (out_shape.dims == 3) out_shape_packed = Mat(out_shape.w, out_shape.h, out_shape.c / out_elempack, (void*)0, out_elemsize, out_elempack);

    // check blob shape
    if (!vkdev->shape_support_image_storage(shape_packed) || !vkdev->shape_support_image_storage(out_shape_packed))
    {
        support_image_storage = false;
        opt.use_image_storage = false;
    }

    if (resize_type == 1 || resize_type == 2)
    {
        std::vector<vk_specialization_type> specializations(2 + 10);
        specializations[0].i = resize_type;
        specializations[1].i = align_corner;
        specializations[2 + 0].i = shape_packed.dims;
        specializations[2 + 1].i = shape_packed.w;
        specializations[2 + 2].i = shape_packed.h;
        specializations[2 + 3].i = shape_packed.c;
        specializations[2 + 4].i = shape_packed.cstep;
        specializations[2 + 5].i = out_shape_packed.dims;
        specializations[2 + 6].i = out_shape_packed.w;
        specializations[2 + 7].i = out_shape_packed.h;
        specializations[2 + 8].i = out_shape_packed.c;
        specializations[2 + 9].i = out_shape_packed.cstep;

        Mat local_size_xyz;
        if (out_shape_packed.dims == 2)
        {
            local_size_xyz.w = std::min(8, out_shape_packed.w);
            local_size_xyz.h = std::min(8, out_shape_packed.h);
            local_size_xyz.c = 1;
        }
        if (out_shape_packed.dims == 3)
        {
            local_size_xyz.w = std::min(4, out_shape_packed.w);
            local_size_xyz.h = std::min(4, out_shape_packed.h);
            local_size_xyz.c = std::min(4, out_shape_packed.c);
        }

        // pack1
        if (shape.dims == 0 || elempack == 1)
        {
            pipeline_interp = new Pipeline(vkdev);
            pipeline_interp->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_interp->create(LayerShaderType::interp, opt, specializations);
        }

        // pack4
        if (shape.dims == 0 || elempack == 4)
        {
            pipeline_interp_pack4 = new Pipeline(vkdev);
            pipeline_interp_pack4->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_interp_pack4->create(LayerShaderType::interp_pack4, opt, specializations);
        }

        // pack8
        if ((opt.use_shader_pack8 && shape.dims == 0) || elempack == 8)
        {
            pipeline_interp_pack8 = new Pipeline(vkdev);
            pipeline_interp_pack8->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_interp_pack8->create(LayerShaderType::interp_pack8, opt, specializations);
        }
    }

    if (resize_type == 3)
    {
        {
            std::vector<vk_specialization_type> specializations(1 + 2);
            specializations[0].i = align_corner;
            specializations[1 + 0].i = shape_packed.w;
            specializations[1 + 1].i = out_shape_packed.w;

            Mat local_size_xyz(64, 1, 1, (void*)0);
            if (out_shape_packed.dims != 0)
            {
                local_size_xyz.w = std::min(64, out_shape_packed.w);
                local_size_xyz.h = 1;
                local_size_xyz.c = 1;
            }

            pipeline_interp_bicubic_coeffs_x = new Pipeline(vkdev);
            pipeline_interp_bicubic_coeffs_x->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_interp_bicubic_coeffs_x->create(LayerShaderType::interp_bicubic_coeffs, opt, specializations);
        }
        {
            std::vector<vk_specialization_type> specializations(1 + 2);
            specializations[0].i = align_corner;
            specializations[1 + 0].i = shape_packed.h;
            specializations[1 + 1].i = out_shape_packed.h;

            Mat local_size_xyz(64, 1, 1, (void*)0);
            if (out_shape_packed.dims != 0)
            {
                local_size_xyz.w = std::min(64, out_shape_packed.h);
                local_size_xyz.h = 1;
                local_size_xyz.c = 1;
            }

            pipeline_interp_bicubic_coeffs_y = new Pipeline(vkdev);
            pipeline_interp_bicubic_coeffs_y->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_interp_bicubic_coeffs_y->create(LayerShaderType::interp_bicubic_coeffs, opt, specializations);
        }

        std::vector<vk_specialization_type> specializations(0 + 10);
        specializations[0 + 0].i = shape_packed.dims;
        specializations[0 + 1].i = shape_packed.w;
        specializations[0 + 2].i = shape_packed.h;
        specializations[0 + 3].i = shape_packed.c;
        specializations[0 + 4].i = shape_packed.cstep;
        specializations[0 + 5].i = out_shape_packed.dims;
        specializations[0 + 6].i = out_shape_packed.w;
        specializations[0 + 7].i = out_shape_packed.h;
        specializations[0 + 8].i = out_shape_packed.c;
        specializations[0 + 9].i = out_shape_packed.cstep;

        Mat local_size_xyz;
        if (out_shape_packed.dims == 2)
        {
            local_size_xyz.w = std::min(8, out_shape_packed.w);
            local_size_xyz.h = std::min(8, out_shape_packed.h);
            local_size_xyz.c = 1;
        }
        if (out_shape_packed.dims == 3)
        {
            local_size_xyz.w = std::min(4, out_shape_packed.w);
            local_size_xyz.h = std::min(4, out_shape_packed.h);
            local_size_xyz.c = std::min(4, out_shape_packed.c);
        }

        // pack1
        if (shape.dims == 0 || elempack == 1)
        {
            pipeline_interp_bicubic = new Pipeline(vkdev);
            pipeline_interp_bicubic->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_interp_bicubic->create(LayerShaderType::interp_bicubic, opt, specializations);
        }

        // pack4
        if (shape.dims == 0 || elempack == 4)
        {
            pipeline_interp_bicubic_pack4 = new Pipeline(vkdev);
            pipeline_interp_bicubic_pack4->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_interp_bicubic_pack4->create(LayerShaderType::interp_bicubic_pack4, opt, specializations);
        }

        // pack8
        if ((opt.use_shader_pack8 && shape.dims == 0) || elempack == 8)
        {
            pipeline_interp_bicubic_pack8 = new Pipeline(vkdev);
            pipeline_interp_bicubic_pack8->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_interp_bicubic_pack8->create(LayerShaderType::interp_bicubic_pack8, opt, specializations);
        }
    }

    return 0;
}

int Interp_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_interp;
    pipeline_interp = 0;

    delete pipeline_interp_pack4;
    pipeline_interp_pack4 = 0;

    delete pipeline_interp_pack8;
    pipeline_interp_pack8 = 0;

    delete pipeline_interp_bicubic_coeffs_x;
    pipeline_interp_bicubic_coeffs_x = 0;

    delete pipeline_interp_bicubic_coeffs_y;
    pipeline_interp_bicubic_coeffs_y = 0;

    delete pipeline_interp_bicubic;
    pipeline_interp_bicubic = 0;

    delete pipeline_interp_bicubic_pack4;
    pipeline_interp_bicubic_pack4 = 0;

    delete pipeline_interp_bicubic_pack8;
    pipeline_interp_bicubic_pack8 = 0;

    return 0;
}

int Interp_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;

    int outw = output_width;
    int outh = output_height;
    if (bottom_blob.dims == 1)
    {
        w = 1;
        h = 1;
    }
    if (outw == 0 || outh == 0)
    {
        outw = static_cast<int>(w * width_scale);
        outh = static_cast<int>(h * height_scale);
    }

    VkMat reference_blob;
    reference_blob.w = outw;
    reference_blob.h = outh;

    std::vector<VkMat> bottom_blobs(2);
    bottom_blobs[0] = bottom_blob;
    bottom_blobs[1] = reference_blob;

    std::vector<VkMat> top_blobs(1);

    int ret = forward(bottom_blobs, top_blobs, cmd, opt);

    top_blob = top_blobs[0];

    return ret;
}

int Interp_vulkan::forward(const VkImageMat& bottom_blob, VkImageMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;

    int outw = output_width;
    int outh = output_height;
    if (bottom_blob.dims == 1)
    {
        w = 1;
        h = 1;
    }
    if (outw == 0 || outh == 0)
    {
        outw = static_cast<int>(w * width_scale);
        outh = static_cast<int>(h * height_scale);
    }

    VkImageMat reference_blob;
    reference_blob.w = outw;
    reference_blob.h = outh;

    std::vector<VkImageMat> bottom_blobs(2);
    bottom_blobs[0] = bottom_blob;
    bottom_blobs[1] = reference_blob;

    std::vector<VkImageMat> top_blobs(1);

    int ret = forward(bottom_blobs, top_blobs, cmd, opt);

    top_blob = top_blobs[0];

    return ret;
}

int Interp_vulkan::forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    const VkMat& bottom_blob = bottom_blobs[0];
    const VkMat& reference_blob = bottom_blobs[1];
    VkMat& top_blob = top_blobs[0];

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int outw = reference_blob.w;
    int outh = reference_blob.h;

    if (dims == 1)
    {
        top_blob.create(outw, outh, w, elemsize, elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;

        std::vector<VkMat> bindings(2);
        bindings[0] = bottom_blob;
        bindings[1] = top_blob;

        std::vector<vk_constant_type> constants(12);
        constants[0].i = bottom_blob.dims;
        constants[1].i = bottom_blob.w;
        constants[2].i = bottom_blob.h;
        constants[3].i = bottom_blob.c;
        constants[4].i = bottom_blob.cstep;
        constants[5].i = top_blob.dims;
        constants[6].i = top_blob.w;
        constants[7].i = top_blob.h;
        constants[8].i = top_blob.c;
        constants[9].i = top_blob.cstep;
        constants[10].f = w / (float)outw;
        constants[11].f = h / (float)outh;

        const Pipeline* pipeline = elempack == 8 ? pipeline_interp_pack8
                                   : elempack == 4 ? pipeline_interp_pack4
                                   : pipeline_interp;

        cmd.record_pipeline(pipeline, bindings, constants, top_blob);

        return 0;
    }

    if (dims == 2)
    {
        if (outw == w)
        {
            top_blob = bottom_blob;
            return 0;
        }

        top_blob.create(outw, h, elemsize, elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;

        if (resize_type == 1 || resize_type == 2) // nearest or bilinear
        {
            std::vector<VkMat> bindings(2);
            bindings[0] = bottom_blob;
            bindings[1] = top_blob;

            std::vector<vk_constant_type> constants(12);
            constants[0].i = bottom_blob.dims;
            constants[1].i = bottom_blob.w;
            constants[2].i = bottom_blob.h;
            constants[3].i = bottom_blob.c;
            constants[4].i = bottom_blob.cstep;
            constants[5].i = top_blob.dims;
            constants[6].i = top_blob.w;
            constants[7].i = top_blob.h;
            constants[8].i = top_blob.c;
            constants[9].i = top_blob.cstep;
            constants[10].f = w / (float)outw;
            constants[11].f = 1.f;

            if (resize_type == 2 && align_corner)
            {
                constants[10].f = (w - 1) / (float)(outw - 1);
            }

            const Pipeline* pipeline = elempack == 8 ? pipeline_interp_pack8
                                       : elempack == 4 ? pipeline_interp_pack4
                                       : pipeline_interp;

            cmd.record_pipeline(pipeline, bindings, constants, top_blob);
        }

        if (resize_type == 3) // bicubic
        {
            VkMat alpha(outw, (size_t)(elemsize / elempack * 4), 4, opt.workspace_vkallocator);
            if (alpha.empty())
                return -100;

            VkMat xofs(outw, (size_t)4u, 1, opt.workspace_vkallocator);
            if (xofs.empty())
                return -100;

            {
                std::vector<VkMat> bindings(2);
                bindings[0] = alpha;
                bindings[1] = xofs;

                std::vector<vk_constant_type> constants(3);
                constants[0].i = bottom_blob.w;
                constants[1].i = outw;
                constants[2].f = (float)bottom_blob.w / outw;

                if (align_corner)
                {
                    constants[2].f = (w - 1) / (float)(outw - 1);
                }

                // record
                cmd.record_pipeline(pipeline_interp_bicubic_coeffs_x, bindings, constants, alpha);
            }

            std::vector<VkMat> bindings(6);
            bindings[0] = bottom_blob;
            bindings[1] = top_blob;
            bindings[2] = alpha;
            bindings[3] = xofs;
            bindings[4] = alpha; // dummy
            bindings[5] = xofs;  // dummy

            std::vector<vk_constant_type> constants(10);
            constants[0].i = bottom_blob.dims;
            constants[1].i = bottom_blob.w;
            constants[2].i = bottom_blob.h;
            constants[3].i = bottom_blob.c;
            constants[4].i = bottom_blob.cstep;
            constants[5].i = top_blob.dims;
            constants[6].i = top_blob.w;
            constants[7].i = top_blob.h;
            constants[8].i = top_blob.c;
            constants[9].i = top_blob.cstep;

            const Pipeline* pipeline = elempack == 8 ? pipeline_interp_bicubic_pack8
                                       : elempack == 4 ? pipeline_interp_bicubic_pack4
                                       : pipeline_interp_bicubic;

            cmd.record_pipeline(pipeline, bindings, constants, top_blob);
        }

        return 0;
    }

    if (outw == w && outh == h)
    {
        top_blob = bottom_blob;
        return 0;
    }

    top_blob.create(outw, outh, channels, elemsize, elempack, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    if (resize_type == 1 || resize_type == 2) // nearest or bilinear
    {
        std::vector<VkMat> bindings(2);
        bindings[0] = bottom_blob;
        bindings[1] = top_blob;

        std::vector<vk_constant_type> constants(12);
        constants[0].i = bottom_blob.dims;
        constants[1].i = bottom_blob.w;
        constants[2].i = bottom_blob.h;
        constants[3].i = bottom_blob.c;
        constants[4].i = bottom_blob.cstep;
        constants[5].i = top_blob.dims;
        constants[6].i = top_blob.w;
        constants[7].i = top_blob.h;
        constants[8].i = top_blob.c;
        constants[9].i = top_blob.cstep;
        constants[10].f = w / (float)outw;
        constants[11].f = h / (float)outh;

        if (resize_type == 2 && align_corner)
        {
            constants[10].f = (w - 1) / (float)(outw - 1);
            constants[11].f = (h - 1) / (float)(outh - 1);
        }

        const Pipeline* pipeline = elempack == 8 ? pipeline_interp_pack8
                                   : elempack == 4 ? pipeline_interp_pack4
                                   : pipeline_interp;

        cmd.record_pipeline(pipeline, bindings, constants, top_blob);
    }
    else if (resize_type == 3) // bicubic
    {
        VkMat alpha(outw, (size_t)(elemsize / elempack * 4), 4, opt.workspace_vkallocator);
        if (alpha.empty())
            return -100;

        VkMat xofs(outw, (size_t)4u, 1, opt.workspace_vkallocator);
        if (xofs.empty())
            return -100;

        {
            std::vector<VkMat> bindings(2);
            bindings[0] = alpha;
            bindings[1] = xofs;

            std::vector<vk_constant_type> constants(3);
            constants[0].i = bottom_blob.w;
            constants[1].i = outw;
            constants[2].f = (float)bottom_blob.w / outw;

            if (align_corner)
            {
                constants[2].f = (w - 1) / (float)(outw - 1);
            }

            // record
            cmd.record_pipeline(pipeline_interp_bicubic_coeffs_x, bindings, constants, alpha);
        }

        VkMat beta(outh, (size_t)(elemsize / elempack * 4), 4, opt.workspace_vkallocator);
        if (beta.empty())
            return -100;

        VkMat yofs(outh, (size_t)4u, 1, opt.workspace_vkallocator);
        if (yofs.empty())
            return -100;

        {
            std::vector<VkMat> bindings(2);
            bindings[0] = beta;
            bindings[1] = yofs;

            std::vector<vk_constant_type> constants(3);
            constants[0].i = bottom_blob.h;
            constants[1].i = outh;
            constants[2].f = (float)bottom_blob.h / outh;

            if (align_corner)
            {
                constants[2].f = (h - 1) / (float)(outh - 1);
            }

            // record
            cmd.record_pipeline(pipeline_interp_bicubic_coeffs_y, bindings, constants, beta);
        }

        std::vector<VkMat> bindings(6);
        bindings[0] = bottom_blob;
        bindings[1] = top_blob;
        bindings[2] = alpha;
        bindings[3] = xofs;
        bindings[4] = beta;
        bindings[5] = yofs;

        std::vector<vk_constant_type> constants(10);
        constants[0].i = bottom_blob.dims;
        constants[1].i = bottom_blob.w;
        constants[2].i = bottom_blob.h;
        constants[3].i = bottom_blob.c;
        constants[4].i = bottom_blob.cstep;
        constants[5].i = top_blob.dims;
        constants[6].i = top_blob.w;
        constants[7].i = top_blob.h;
        constants[8].i = top_blob.c;
        constants[9].i = top_blob.cstep;

        const Pipeline* pipeline = elempack == 8 ? pipeline_interp_bicubic_pack8
                                   : elempack == 4 ? pipeline_interp_bicubic_pack4
                                   : pipeline_interp_bicubic;

        cmd.record_pipeline(pipeline, bindings, constants, top_blob);
    }

    return 0;
}

int Interp_vulkan::forward(const std::vector<VkImageMat>& bottom_blobs, std::vector<VkImageMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    const VkImageMat& bottom_blob = bottom_blobs[0];
    const VkImageMat& reference_blob = bottom_blobs[1];
    VkImageMat& top_blob = top_blobs[0];

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int outw = reference_blob.w;
    int outh = reference_blob.h;

    if (dims == 1)
    {
        top_blob.create(outw, outh, w, elemsize, elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;

        std::vector<VkImageMat> bindings(2);
        bindings[0] = bottom_blob;
        bindings[1] = top_blob;

        std::vector<vk_constant_type> constants(12);
        constants[0].i = bottom_blob.dims;
        constants[1].i = bottom_blob.w;
        constants[2].i = bottom_blob.h;
        constants[3].i = bottom_blob.c;
        constants[4].i = 0; //bottom_blob.cstep;
        constants[5].i = top_blob.dims;
        constants[6].i = top_blob.w;
        constants[7].i = top_blob.h;
        constants[8].i = top_blob.c;
        constants[9].i = 0; //top_blob.cstep;
        constants[10].f = w / (float)outw;
        constants[11].f = h / (float)outh;

        const Pipeline* pipeline = elempack == 8 ? pipeline_interp_pack8
                                   : elempack == 4 ? pipeline_interp_pack4
                                   : pipeline_interp;

        cmd.record_pipeline(pipeline, bindings, constants, top_blob);

        return 0;
    }

    if (dims == 2)
    {
        if (outw == w)
        {
            top_blob = bottom_blob;
            return 0;
        }

        top_blob.create(outw, h, elemsize, elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;

        if (resize_type == 1 || resize_type == 2) // nearest or bilinear
        {
            std::vector<VkImageMat> bindings(2);
            bindings[0] = bottom_blob;
            bindings[1] = top_blob;

            std::vector<vk_constant_type> constants(12);
            constants[0].i = bottom_blob.dims;
            constants[1].i = bottom_blob.w;
            constants[2].i = bottom_blob.h;
            constants[3].i = bottom_blob.c;
            constants[4].i = 0; //bottom_blob.cstep;
            constants[5].i = top_blob.dims;
            constants[6].i = top_blob.w;
            constants[7].i = top_blob.h;
            constants[8].i = top_blob.c;
            constants[9].i = 0; //top_blob.cstep;
            constants[10].f = w / (float)outw;
            constants[11].f = 1.f;

            if (resize_type == 2 && align_corner)
            {
                constants[10].f = (w - 1) / (float)(outw - 1);
            }

            const Pipeline* pipeline = elempack == 8 ? pipeline_interp_pack8
                                       : elempack == 4 ? pipeline_interp_pack4
                                       : pipeline_interp;

            cmd.record_pipeline(pipeline, bindings, constants, top_blob);
        }

        if (resize_type == 3) // bicubic
        {
            VkMat alpha(outw, (size_t)(elemsize / elempack * 4), 4, opt.workspace_vkallocator);
            if (alpha.empty())
                return -100;

            VkMat xofs(outw, (size_t)4u, 1, opt.workspace_vkallocator);
            if (xofs.empty())
                return -100;

            {
                std::vector<VkMat> bindings(2);
                bindings[0] = alpha;
                bindings[1] = xofs;

                std::vector<vk_constant_type> constants(3);
                constants[0].i = bottom_blob.w;
                constants[1].i = outw;
                constants[2].f = (float)bottom_blob.w / outw;

                if (align_corner)
                {
                    constants[2].f = (w - 1) / (float)(outw - 1);
                }

                // record
                cmd.record_pipeline(pipeline_interp_bicubic_coeffs_x, bindings, constants, alpha);
            }

            std::vector<VkMat> buffer_bindings(4);
            buffer_bindings[0] = alpha;
            buffer_bindings[1] = xofs;

            std::vector<VkImageMat> image_bindings(2);
            image_bindings[0] = bottom_blob;
            image_bindings[1] = top_blob;

            std::vector<vk_constant_type> constants(10);
            constants[0].i = bottom_blob.dims;
            constants[1].i = bottom_blob.w;
            constants[2].i = bottom_blob.h;
            constants[3].i = bottom_blob.c;
            constants[4].i = 0; //bottom_blob.cstep;
            constants[5].i = top_blob.dims;
            constants[6].i = top_blob.w;
            constants[7].i = top_blob.h;
            constants[8].i = top_blob.c;
            constants[9].i = 0; //top_blob.cstep;

            const Pipeline* pipeline = elempack == 8 ? pipeline_interp_bicubic_pack8
                                       : elempack == 4 ? pipeline_interp_bicubic_pack4
                                       : pipeline_interp_bicubic;

            cmd.record_pipeline(pipeline, buffer_bindings, image_bindings, constants, top_blob);
        }

        return 0;
    }

    if (outw == w && outh == h)
    {
        top_blob = bottom_blob;
        return 0;
    }

    top_blob.create(outw, outh, channels, elemsize, elempack, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    if (resize_type == 1 || resize_type == 2) // nearest or bilinear
    {
        std::vector<VkImageMat> bindings(2);
        bindings[0] = bottom_blob;
        bindings[1] = top_blob;

        std::vector<vk_constant_type> constants(12);
        constants[0].i = bottom_blob.dims;
        constants[1].i = bottom_blob.w;
        constants[2].i = bottom_blob.h;
        constants[3].i = bottom_blob.c;
        constants[4].i = 0; //bottom_blob.cstep;
        constants[5].i = top_blob.dims;
        constants[6].i = top_blob.w;
        constants[7].i = top_blob.h;
        constants[8].i = top_blob.c;
        constants[9].i = 0; //top_blob.cstep;
        constants[10].f = w / (float)outw;
        constants[11].f = h / (float)outh;

        if (resize_type == 2 && align_corner)
        {
            constants[10].f = (w - 1) / (float)(outw - 1);
            constants[11].f = (h - 1) / (float)(outh - 1);
        }

        const Pipeline* pipeline = elempack == 8 ? pipeline_interp_pack8
                                   : elempack == 4 ? pipeline_interp_pack4
                                   : pipeline_interp;

        cmd.record_pipeline(pipeline, bindings, constants, top_blob);
    }
    else if (resize_type == 3) // bicubic
    {
        VkMat alpha(outw, (size_t)(elemsize / elempack * 4), 4, opt.workspace_vkallocator);
        if (alpha.empty())
            return -100;

        VkMat xofs(outw, (size_t)4u, 1, opt.workspace_vkallocator);
        if (xofs.empty())
            return -100;

        {
            std::vector<VkMat> bindings(2);
            bindings[0] = alpha;
            bindings[1] = xofs;

            std::vector<vk_constant_type> constants(3);
            constants[0].i = bottom_blob.w;
            constants[1].i = outw;
            constants[2].f = (float)bottom_blob.w / outw;

            if (align_corner)
            {
                constants[2].f = (w - 1) / (float)(outw - 1);
            }

            // record
            cmd.record_pipeline(pipeline_interp_bicubic_coeffs_x, bindings, constants, alpha);
        }

        VkMat beta(outh, (size_t)(elemsize / elempack * 4), 4, opt.workspace_vkallocator);
        if (beta.empty())
            return -100;

        VkMat yofs(outh, (size_t)4u, 1, opt.workspace_vkallocator);
        if (yofs.empty())
            return -100;

        {
            std::vector<VkMat> bindings(2);
            bindings[0] = beta;
            bindings[1] = yofs;

            std::vector<vk_constant_type> constants(3);
            constants[0].i = bottom_blob.h;
            constants[1].i = outh;
            constants[2].f = (float)bottom_blob.h / outh;

            if (align_corner)
            {
                constants[2].f = (h - 1) / (float)(outh - 1);
            }

            // record
            cmd.record_pipeline(pipeline_interp_bicubic_coeffs_y, bindings, constants, beta);
        }

        std::vector<VkMat> buffer_bindings(4);
        buffer_bindings[0] = alpha;
        buffer_bindings[1] = xofs;
        buffer_bindings[2] = beta;
        buffer_bindings[3] = yofs;

        std::vector<VkImageMat> image_bindings(2);
        image_bindings[0] = bottom_blob;
        image_bindings[1] = top_blob;

        std::vector<vk_constant_type> constants(10);
        constants[0].i = bottom_blob.dims;
        constants[1].i = bottom_blob.w;
        constants[2].i = bottom_blob.h;
        constants[3].i = bottom_blob.c;
        constants[4].i = 0; //bottom_blob.cstep;
        constants[5].i = top_blob.dims;
        constants[6].i = top_blob.w;
        constants[7].i = top_blob.h;
        constants[8].i = top_blob.c;
        constants[9].i = 0; //top_blob.cstep;

        const Pipeline* pipeline = elempack == 8 ? pipeline_interp_bicubic_pack8
                                   : elempack == 4 ? pipeline_interp_bicubic_pack4
                                   : pipeline_interp_bicubic;

        cmd.record_pipeline(pipeline, buffer_bindings, image_bindings, constants, top_blob);
    }

    return 0;
}

} // namespace ncnn
