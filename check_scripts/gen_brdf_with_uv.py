import os 
import cv2
import OpenEXR as openexr

import utils.file_util as futil

exr_path = 'renderings/check_renderings/Skull01_001_001_bl.exr'

if __name__ == '__main__':
    exr_file = openexr.InputFile(exr_path)
    print(exr_file.header())
    dw = exr_file.header()['dataWindow']
    size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
    uv_a = futil.get_exr_cmpnt(exr_file, 'ViewLayer.UV.A', size)
    uv_u = futil.get_exr_cmpnt(exr_file, 'ViewLayer.UV.U', size)
    uv_v = futil.get_exr_cmpnt(exr_file, 'ViewLayer.UV.V', size)
    print(uv_a.min(),uv_a.max(), uv_a.shape,'\n')
    print(uv_u.min(),uv_u.max(), uv_u.shape,'\n')
    print(uv_v.min(),uv_v.max(), uv_v.shape,'\n')
    

    albedo_path = 'tmp/tmp_1/Skull01/albedo_diff.png'
    roughness_path = 'tmp/tmp_1/Skull01/roughness.png'
    albedo_bake_map = cv2.imread(albedo_path, -1)
    roughness_bake_map = cv2.imread(roughness_path, -1)
    uv_u, uv_v = (uv_u)*(albedo_bake_map.shape[1]-1), (1-uv_v)*(albedo_bake_map.shape[0]-1)
    print(uv_u[342, 420], uv_v[342, 420], )
    print(albedo_bake_map[1813,1192,],albedo_bake_map[1813,1193,])
    print(albedo_bake_map[1814,1192,],albedo_bake_map[1814,1193,])
    albedo = cv2.remap(albedo_bake_map, uv_u, uv_v, cv2.INTER_LINEAR)*uv_a[...,None]
    roughness = cv2.remap(roughness_bake_map, uv_u, uv_v, cv2.INTER_LINEAR)*uv_a


    cv2.imwrite('albedo.png', (albedo/255.)**(2.2)*255)
    cv2.imwrite('roughness.png', roughness)
