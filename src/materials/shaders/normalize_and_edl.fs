
#extension GL_EXT_frag_depth : enable

precision mediump float;
precision mediump int;

uniform sampler2D uWeightMap;
uniform sampler2D uEDLMap;
uniform sampler2D uDepthMap;

uniform float screenWidth;
uniform float screenHeight;
uniform vec2 neighbours[NEIGHBOUR_COUNT];
uniform float edlStrength;
uniform float radius;

varying vec2 vUv;

float response(float depth){
	vec2 uvRadius = radius / vec2(screenWidth, screenHeight);
	
	float sum = 0.0;
	
	for(int i = 0; i < NEIGHBOUR_COUNT; i++){
		vec2 uvNeighbor = vUv + uvRadius * neighbours[i];
		
		float neighbourDepth = texture2D(uEDLMap, uvNeighbor).a;

		if(neighbourDepth != 0.0){
			if(depth == 0.0){
				sum += 100.0;
			}else{
				sum += max(0.0, depth - neighbourDepth);
			}
		}
	}
	
	return sum / float(NEIGHBOUR_COUNT);
}



// Iconem Add normal_from_depth_buffer_texture 
// TODO : DEPTH BUFFER IS ONLY USED WITHIN EDL AND NORMALIZATION MATERIALS, not a simple pointcloudmaterial. And neither have near and far values to correctly scale normals
float ztransform(vec2 fragPos){
	float uNear = 0.001, uFar = 1000.;
    // Use in EDL shader : uEDLDepth // in HQ normalize_and_edl : uHQDepthMap
    float z_n = 2.0 * texture2D(uDepthMap, fragPos).r - 1.0; //z_n between -1 and 1
    return 2. * uFar * uNear / (uFar + uNear - (uFar - uNear) * z_n ); //z_out between uNear and uFar
}
vec3 getNormalFromDepth(vec2 fragPos) {
    vec2 offset1 = vec2(0.0, 1./screenWidth);
    vec2 offset2 = vec2(1./screenHeight, 0.0);

    float depth = ztransform(fragPos);
    float depth1 = ztransform(fragPos + offset1);
    float depth2 = ztransform(fragPos + offset2);
    vec3 p1 = vec3(offset1, depth1 - depth);
    vec3 p2 = vec3(offset2, depth2 - depth);
    vec3 normal = cross(p1, p2);
    normal.z = -normal.z * depth;
    normal = normalize(normal);
    return normal * 0.5 + 0.5;    
}

void main() {

	float edlDepth = texture2D(uEDLMap, vUv).a;
	float res = response(edlDepth);
	float shade = exp(-res * 300.0 * edlStrength);

	float depth = texture2D(uDepthMap, vUv).r;
	if(depth >= 1.0 && res == 0.0){
		discard;
	}
	
	vec4 color = texture2D(uWeightMap, vUv); 
	color = color / color.w;
	color = color * shade;

	gl_FragColor = vec4(color.xyz, 1.0); 
	gl_FragColor = vec4(getNormalFromDepth(vUv), 1.);

	gl_FragDepthEXT = depth;
}