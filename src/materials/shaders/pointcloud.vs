
precision highp float;
precision highp int;

#define max_clip_polygons 8
#define PI 3.141592653589793

attribute vec3 position;
attribute vec3 color;
attribute float intensity;
attribute float classification;
attribute float returnNumber;
attribute float numberOfReturns;
attribute float pointSourceID;
attribute vec4 indices;
attribute float spacing;
attribute float gpsTime;

uniform mat4 modelMatrix;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;
uniform mat4 uViewInv;

uniform float uScreenWidth;
uniform float uScreenHeight;
uniform float fov;
uniform float near;
uniform float far;

uniform bool uDebug;

uniform bool uUseOrthographicCamera;
uniform float uOrthoWidth;
uniform float uOrthoHeight;

// downsampling density based on parameters
#if defined(num_downsamplingPolygonVerts)
uniform float downsamplingWidth;
uniform mat4 modelMatrixInverse;
#endif
// if num_downsamplingPolygonVerts == -1: no downsampling, if == 0: sphere if > 0: polygon
#if defined(num_downsamplingPolygonVerts) && num_downsamplingPolygonVerts > 0
uniform vec3 downsamplingPolygonVerts[num_downsamplingPolygonVerts];
#elif defined(num_downsamplingPolygonVerts) && num_downsamplingPolygonVerts == 0
uniform mat4 downsamplingEllipseMatrix;
uniform mat4 downsamplingEllipseMatrixInverse;
uniform mat4 downsamplingEllipseMatrixInverse_modelMatrix;
//uniform vec3 downsamplingPolygonVerts[100];
#endif


#define CLIPTASK_NONE 0
#define CLIPTASK_HIGHLIGHT 1
#define CLIPTASK_SHOW_INSIDE 2
#define CLIPTASK_SHOW_OUTSIDE 3

#define CLIPMETHOD_INSIDE_ANY 0
#define CLIPMETHOD_INSIDE_ALL 1

uniform int clipTask;
uniform int clipMethod;
#if defined(num_clipboxes) && num_clipboxes > 0
	uniform mat4 clipBoxes[num_clipboxes];
#endif

#if defined(num_clipspheres) && num_clipspheres > 0
	uniform mat4 uClipSpheres[num_clipspheres];
#endif

#if defined(num_clippolygons) && num_clippolygons > 0
	uniform int uClipPolygonVCount[num_clippolygons];
	uniform vec3 uClipPolygonVertices[num_clippolygons * 8];
	uniform mat4 uClipPolygonWVP[num_clippolygons];
#endif


uniform float size;
uniform float minSize;
uniform float maxSize;

uniform float uPCIndex;
uniform float uOctreeSpacing;
uniform float uNodeSpacing;
uniform float uOctreeSize;
uniform vec3 uBBSize;
uniform float uLevel;
uniform float uVNStart;
uniform bool uIsLeafNode;

uniform vec3 uColor;
uniform float uOpacity;

uniform vec2 elevationRange;
uniform vec2 intensityRange;

uniform vec2 uFilterReturnNumberRange;
uniform vec2 uFilterNumberOfReturnsRange;
uniform vec2 uFilterGPSTimeClipRange;

uniform float uGPSOffset;
uniform float uGPSRange;
uniform float intensityGamma;
uniform float intensityContrast;
uniform float intensityBrightness;
uniform float rgbGamma;
uniform float rgbContrast;
uniform float rgbBrightness;
uniform float uTransition;
uniform float wRGB;
uniform float wIntensity;
uniform float wElevation;
uniform float wClassification;
uniform float wReturnNumber;
uniform float wSourceID;

uniform vec3 uShadowColor;

uniform sampler2D visibleNodes;
uniform sampler2D gradient;
uniform sampler2D classificationLUT;

#if defined(num_shadowmaps) && num_shadowmaps > 0
uniform sampler2D uShadowMap[num_shadowmaps];
uniform mat4 uShadowWorldView[num_shadowmaps];
uniform mat4 uShadowProj[num_shadowmaps];
#endif

varying vec3	vColor;
varying float	vLogDepth;
varying vec3	vViewPosition;
varying float 	vRadius;
varying float 	vPointSize;


float round(float number){
	return floor(number + 0.5);
}

// 
//    ###    ########     ###    ########  ######## #### ##     ## ########     ######  #### ######## ########  ######  
//   ## ##   ##     ##   ## ##   ##     ##    ##     ##  ##     ## ##          ##    ##  ##       ##  ##       ##    ## 
//  ##   ##  ##     ##  ##   ##  ##     ##    ##     ##  ##     ## ##          ##        ##      ##   ##       ##       
// ##     ## ##     ## ##     ## ########     ##     ##  ##     ## ######       ######   ##     ##    ######    ######  
// ######### ##     ## ######### ##           ##     ##   ##   ##  ##                ##  ##    ##     ##             ## 
// ##     ## ##     ## ##     ## ##           ##     ##    ## ##   ##          ##    ##  ##   ##      ##       ##    ## 
// ##     ## ########  ##     ## ##           ##    ####    ###    ########     ######  #### ######## ########  ######  
// 																			


// ---------------------
// OCTREE
// ---------------------

#if (defined(adaptive_point_size) || defined(color_type_lod)) && defined(tree_type_octree)
/**
 * number of 1-bits up to inclusive index position
 * number is treated as if it were an integer in the range 0-255
 *
 */
int numberOfOnes(int number, int index){
	int numOnes = 0;
	int tmp = 128;
	for(int i = 7; i >= 0; i--){
		
		if(number >= tmp){
			number = number - tmp;

			if(i <= index){
				numOnes++;
			}
		}
		
		tmp = tmp / 2;
	}

	return numOnes;
}


/**
 * checks whether the bit at index is 1
 * number is treated as if it were an integer in the range 0-255
 *
 */
bool isBitSet(int number, int index){

	// weird multi else if due to lack of proper array, int and bitwise support in WebGL 1.0
	int powi = 1;
	if(index == 0){
		powi = 1;
	}else if(index == 1){
		powi = 2;
	}else if(index == 2){
		powi = 4;
	}else if(index == 3){
		powi = 8;
	}else if(index == 4){
		powi = 16;
	}else if(index == 5){
		powi = 32;
	}else if(index == 6){
		powi = 64;
	}else if(index == 7){
		powi = 128;
	}else{
		return false;
	}

	int ndp = number / powi;

	return mod(float(ndp), 2.0) != 0.0;
}


/**
 * find the LOD at the point position
 */
float getLOD(){
	
	vec3 offset = vec3(0.0, 0.0, 0.0);
	int iOffset = int(uVNStart);
	float depth = uLevel;
	for(float i = 0.0; i <= 30.0; i++){
		float nodeSizeAtLevel = uOctreeSize  / pow(2.0, i + uLevel + 0.0);
		
		vec3 index3d = (position-offset) / nodeSizeAtLevel;
		index3d = floor(index3d + 0.5);
		int index = int(round(4.0 * index3d.x + 2.0 * index3d.y + index3d.z));
		
		vec4 value = texture2D(visibleNodes, vec2(float(iOffset) / 2048.0, 0.0));
		int mask = int(round(value.r * 255.0));

		if(isBitSet(mask, index)){
			// there are more visible child nodes at this position
			int advanceG = int(round(value.g * 255.0)) * 256;
			int advanceB = int(round(value.b * 255.0));
			int advanceChild = numberOfOnes(mask, index - 1);
			int advance = advanceG + advanceB + advanceChild;

			iOffset = iOffset + advance;
			
			depth++;
		}else{
			// no more visible child nodes at this position
			return value.a * 255.0;
			//return depth;
		}
		
		offset = offset + (vec3(1.0, 1.0, 1.0) * nodeSizeAtLevel * 0.5) * index3d;
	}
		
	return depth;
}

float getSpacing(){
	vec3 offset = vec3(0.0, 0.0, 0.0);
	int iOffset = int(uVNStart);
	float depth = uLevel;
	float spacing = uNodeSpacing;
	for(float i = 0.0; i <= 30.0; i++){
		float nodeSizeAtLevel = uOctreeSize  / pow(2.0, i + uLevel + 0.0);
		
		vec3 index3d = (position-offset) / nodeSizeAtLevel;
		index3d = floor(index3d + 0.5);
		int index = int(round(4.0 * index3d.x + 2.0 * index3d.y + index3d.z));
		
		vec4 value = texture2D(visibleNodes, vec2(float(iOffset) / 2048.0, 0.0));
		int mask = int(round(value.r * 255.0));
		float spacingFactor = value.a;

		if(i > 0.0){
			spacing = spacing / (255.0 * spacingFactor);
		}
		

		if(isBitSet(mask, index)){
			// there are more visible child nodes at this position
			int advanceG = int(round(value.g * 255.0)) * 256;
			int advanceB = int(round(value.b * 255.0));
			int advanceChild = numberOfOnes(mask, index - 1);
			int advance = advanceG + advanceB + advanceChild;

			iOffset = iOffset + advance;

			//spacing = spacing / (255.0 * spacingFactor);
			//spacing = spacing / 3.0;
			
			depth++;
		}else{
			// no more visible child nodes at this position
			return spacing;
		}
		
		offset = offset + (vec3(1.0, 1.0, 1.0) * nodeSizeAtLevel * 0.5) * index3d;
	}
		
	return spacing;
}

float getPointSizeAttenuation(){
	return pow(2.0, getLOD());
}


#endif


// ---------------------
// KD-TREE
// ---------------------

#if (defined(adaptive_point_size) || defined(color_type_lod)) && defined(tree_type_kdtree)

float getLOD(){
	vec3 offset = vec3(0.0, 0.0, 0.0);
	float iOffset = 0.0;
	float depth = 0.0;
		
		
	vec3 size = uBBSize;	
	vec3 pos = position;
		
	for(float i = 0.0; i <= 1000.0; i++){
		
		vec4 value = texture2D(visibleNodes, vec2(iOffset / 2048.0, 0.0));
		
		int children = int(value.r * 255.0);
		float next = value.g * 255.0;
		int split = int(value.b * 255.0);
		
		if(next == 0.0){
		 	return depth;
		}
		
		vec3 splitv = vec3(0.0, 0.0, 0.0);
		if(split == 1){
			splitv.x = 1.0;
		}else if(split == 2){
		 	splitv.y = 1.0;
		}else if(split == 4){
		 	splitv.z = 1.0;
		}
		
		iOffset = iOffset + next;
		
		float factor = length(pos * splitv / size);
		if(factor < 0.5){
			// left
		if(children == 0 || children == 2){
				return depth;
			}
		}else{
			// right
			pos = pos - size * splitv * 0.5;
			if(children == 0 || children == 1){
				return depth;
			}
			if(children == 3){
				iOffset = iOffset + 1.0;
			}
		}
		size = size * ((1.0 - (splitv + 1.0) / 2.0) + 0.5);
		
		depth++;
	}
		
		
	return depth;	
}

float getPointSizeAttenuation(){
	return 0.5 * pow(1.3, getLOD());
}

#endif



// 
//    ###    ######## ######## ########  #### ########  ##     ## ######## ########  ######  
//   ## ##      ##       ##    ##     ##  ##  ##     ## ##     ##    ##    ##       ##    ## 
//  ##   ##     ##       ##    ##     ##  ##  ##     ## ##     ##    ##    ##       ##       
// ##     ##    ##       ##    ########   ##  ########  ##     ##    ##    ######    ######  
// #########    ##       ##    ##   ##    ##  ##     ## ##     ##    ##    ##             ## 
// ##     ##    ##       ##    ##    ##   ##  ##     ## ##     ##    ##    ##       ##    ## 
// ##     ##    ##       ##    ##     ## #### ########   #######     ##    ########  ######                                                                               
// 



// formula adapted from: http://www.dfstudios.co.uk/articles/programming/image-programming-algorithms/image-processing-algorithms-part-5-contrast-adjustment/
float getContrastFactor(float contrast){
	return (1.0158730158730156 * (contrast + 1.0)) / (1.0158730158730156 - contrast);
}

vec3 getRGB(){
	vec3 rgb = color;
	
	rgb = pow(rgb, vec3(rgbGamma));
	rgb = rgb + rgbBrightness;
	//rgb = (rgb - 0.5) * getContrastFactor(rgbContrast) + 0.5;
	rgb = clamp(rgb, 0.0, 1.0);

		//rgb = indices.rgb;
	//rgb.b = pcIndex / 255.0;
	
	
	return rgb;
}

float getIntensity(){
	float w = (intensity - intensityRange.x) / (intensityRange.y - intensityRange.x);
	w = pow(w, intensityGamma);
	w = w + intensityBrightness;
	w = (w - 0.5) * getContrastFactor(intensityContrast) + 0.5;
	w = clamp(w, 0.0, 1.0);

	return w;
}

float getGpsTime(){
	float w = (gpsTime + uGPSOffset) / uGPSRange;
	w = clamp(w, 0.0, 1.0);

	return w;
}

vec3 getElevation(){
	vec4 world = modelMatrix * vec4( position, 1.0 );
	float w = (world.z - elevationRange.x) / (elevationRange.y - elevationRange.x);
	vec3 cElevation = texture2D(gradient, vec2(w,1.0-w)).rgb;
	
	return cElevation;
}

vec4 getClassification(){
	vec2 uv = vec2(classification / 255.0, 0.5);
	vec4 classColor = texture2D(classificationLUT, uv);
	
	return classColor;
}

vec3 getReturnNumber(){
	if(numberOfReturns == 1.0){
		return vec3(1.0, 1.0, 0.0);
	}else{
		if(returnNumber == 1.0){
			return vec3(1.0, 0.0, 0.0);
		}else if(returnNumber == numberOfReturns){
			return vec3(0.0, 0.0, 1.0);
		}else{
			return vec3(0.0, 1.0, 0.0);
		}
	}
}

vec3 getSourceID(){
	float w = mod(pointSourceID, 10.0) / 10.0;
	return texture2D(gradient, vec2(w,1.0 - w)).rgb;
}

vec3 getCompositeColor(){
	vec3 c;
	float w;

	c += wRGB * getRGB();
	w += wRGB;
	
	c += wIntensity * getIntensity() * vec3(1.0, 1.0, 1.0);
	w += wIntensity;
	
	c += wElevation * getElevation();
	w += wElevation;
	
	c += wReturnNumber * getReturnNumber();
	w += wReturnNumber;
	
	c += wSourceID * getSourceID();
	w += wSourceID;
	
	vec4 cl = wClassification * getClassification();
    c += cl.a * cl.rgb;
	w += wClassification * cl.a;

	c = c / w;
	
	if(w == 0.0){
		//c = color;
		gl_Position = vec4(100.0, 100.0, 100.0, 0.0);
	}
	
	return c;
}


float rand_f(vec3 pos){ 
	return fract(sin(pos.z * dot(pos.xy, vec2(12.9898,78.233))) * 43758.5453); 
} 


float getDistance_Ellipsoid(){ 

#if defined(num_downsamplingPolygonVerts) && num_downsamplingPolygonVerts == 0
	// viewer.scene.pointclouds[0].material.downsamplingEllipseMatrix = viewer.scene.volumes[0].matrixWorld.clone()

    vec4 pos_world = modelMatrix * vec4( position, 1.0 );
	float pointDist_local = length((downsamplingEllipseMatrixInverse * pos_world).xyz); // faster approximate, but then hard to scale correctly if sx sy sz different
	vec3 closest_point_unit_sphere = 1. * normalize((downsamplingEllipseMatrixInverse * pos_world).xyz);
	float pointDist = length((modelMatrixInverse * downsamplingEllipseMatrix * vec4(closest_point_unit_sphere, 1.)).xyz - position); 

	// vec4 downsamplingPoint = vec4(downsamplingEllipseMatrix[3][0], downsamplingEllipseMatrix[3][1], downsamplingEllipseMatrix[3][2], 1.);
    // vec3 pos_local = vec3(modelMatrixInverse * downsamplingPoint );
	// pointDist = length(pos_world - downsamplingPoint); 
    // pointDist = length(position - pos_local); 

	//float dist_normalized = (pointDist - min_dist) / (max_dist - min_dist); // dist between 0 and 1 
	// float dist_normalized = (pointDist - 1.) / downsamplingWidth; // dist between 0 and 1 
	float dist = pointDist_local <= 1. ? 0. : pointDist; // dist between 0 and 1 
#else
	float dist = 0.;
#endif
	return dist; 
} 

// https://wrf.ecse.rpi.edu/Research/Short_Notes/pnpoly.html
// cannot find correct definition, so embed in shader code due to verts signature
/*
int ptInPoly(int nvert, vec3 verts[100], vec3 pt)
{
  int i, j, c = 0;
  j = nvert - 1;
  for (i = 0; i < nvert; i++) {
	j = i;
    if ( ((verts[i].y > pt.y) != (verts[j].y > pt.y)) &&
     (pt.x < (verts[j].x - verts[i].x) * (pt.y - verts[i].y) / (verts[j].y - verts[i].y) + verts[i].x) )
       c = 1 - c;
  }
  return c;
}
*/

float getDistance_Polygon(){ 
	vec4 pos_world = modelMatrix * vec4( position, 1.0 );
	float dist = -1.; 
#if defined(num_downsamplingPolygonVerts) && num_downsamplingPolygonVerts > 0
	for (int i = 0; i < 0+1 * (num_downsamplingPolygonVerts + 1); i++) {
		vec4 e0_ = modelMatrixInverse * vec4(downsamplingPolygonVerts[i], 1);
		vec4 e1_ = modelMatrixInverse * vec4(downsamplingPolygonVerts[i + 1 < num_downsamplingPolygonVerts ? i + 1 : 0], 1);
		vec2 e0 = vec2((e0_ / e0_.w ).x, (e0_ / e0_.w ).y);
		vec2 e1 = vec2((e1_ / e1_.w ).x, (e1_ / e1_.w ).y);
		//vec3 e1 = (modelMatrixInverse * vec4(downsamplingPolygonVerts[(i + 1)], 1) ).xyz;
		vec2 e = e1 - e0;
		vec2 pos_xy = vec2(position.x, position.y);
		vec2 v = pos_xy - e0;
		float s = dot(v, e) / ( length(e) *  length(e)); 
		vec2 closest;
		float dist_i;
		if (s <= 0.) {
			closest = e0;
		} else if (s <= 1.) {
			closest = e0 + s * e;
		} else {
			closest = e1;
		}
		dist_i = length(pos_xy - closest);
		//if (cross(e, v).z > 0.) {
		/*
		if (e.x * v.y - e.y * v.x > 0.) {
			dist_i = 0.;
		}*/
		//dist = min(dist_i, dist);

		// Work with clockwise or counter-cw polygons;
		dist_i = abs(dist_i);
		
		// distance is minimum of two distances, unless point is at least on one side inside polygon
		if (i == 0) {
			dist = dist_i;
		}
		if (abs(dist_i) < 0.001 || abs(dist) < 0.001) {
			dist = max(dist_i, dist);
		} else {
			dist = min(dist_i, dist);
		}
	}
	
	// Check if point is inside polygon, concave/convex. 
	// Working with concave polygons while test on cross product of v and e is only working on convex
	int j; 
	bool ptInPoly = false;
	j = num_downsamplingPolygonVerts - 1;
	for (int i = 0; i < num_downsamplingPolygonVerts; i++) {
		j =  (i > 0) ? (i - 1) : num_downsamplingPolygonVerts - 1;
		if ( ((downsamplingPolygonVerts[i].y > pos_world.y) != (downsamplingPolygonVerts[j].y > pos_world.y)) &&
		(pos_world.x < (downsamplingPolygonVerts[j].x - downsamplingPolygonVerts[i].x) * (pos_world.y - downsamplingPolygonVerts[i].y) / (downsamplingPolygonVerts[j].y - downsamplingPolygonVerts[i].y) + downsamplingPolygonVerts[i].x) )
		{ptInPoly = !ptInPoly;}
	}
	if (ptInPoly) {
		dist = 0.;
	} 
#endif

	return dist; 
} 


float getNormalizedDistance(){ 
	float dist; 
#if defined(num_downsamplingPolygonVerts) && num_downsamplingPolygonVerts >= 0
	if (num_downsamplingPolygonVerts > 0) {
		dist = getDistance_Polygon();
	} else {
		dist = getDistance_Ellipsoid(); 
	}
	float dist_normalized = dist / downsamplingWidth;
	// TRUNCATED LINEAR, non soft changes in density
	// dist_normalized = min(max(dist_normalized, 0.), 1.); //truncated linear
	// SIGMOID, smoother
	dist_normalized = 1. / (1. + exp (- 1. * (dist - 0.*downsamplingWidth) / downsamplingWidth ));
	// OTHER EXAMPLE SHADER FUNCTIONS : http://www.flong.com/texts/code/shapers_exp/
	// Display 1 point out of 1000 on the outer boundary, might need to be removed
	dist_normalized -= 0.001;
#else
	float dist_normalized = 0.;
#endif
	return dist_normalized;
}


float getDownsamplingVisibility(){ 
    float dist_normalized = getNormalizedDistance();// * 0.8 + 0.1*sin(uTime * 2.); 
    float rand_n = rand_f(position.xyz);  
    return ((min(max(1. * (dist_normalized - 0.5), -0.5), 0.5) + 0.5) > rand_n ? 0. : 1.);  
} 
  



// 
//  ######  ##       #### ########  ########  #### ##    ##  ######   
// ##    ## ##        ##  ##     ## ##     ##  ##  ###   ## ##    ##  
// ##       ##        ##  ##     ## ##     ##  ##  ####  ## ##        
// ##       ##        ##  ########  ########   ##  ## ## ## ##   #### 
// ##       ##        ##  ##        ##         ##  ##  #### ##    ##  
// ##    ## ##        ##  ##        ##         ##  ##   ### ##    ##  
//  ######  ######## #### ##        ##        #### ##    ##  ######                                                          
// 



vec3 getColor(){
	vec3 color;
	
	#ifdef color_type_rgb
		color = getRGB();
	#elif defined color_type_height
		color = getElevation();
		float w = (getNormalizedDistance() - elevationRange.x) / (elevationRange.y - elevationRange.x);
		color = texture2D(gradient, vec2(w,1.0-w)).rgb;

		color = getNormalizedDistance() * vec3(1., 1., 1.);
	#elif defined color_type_rgb_height
		vec3 cHeight = getElevation();
		color = (1.0 - uTransition) * getRGB() + uTransition * cHeight;
	#elif defined color_type_depth
		float linearDepth = gl_Position.w;
		float expDepth = (gl_Position.z / gl_Position.w) * 0.5 + 0.5;
		color = vec3(linearDepth, expDepth, 0.0);
	#elif defined color_type_intensity
		float w = getIntensity();
		color = vec3(w, w, w);
	#elif defined color_type_gpstime
		float w = getGpsTime();
		color = vec3(w, w, w);
	#elif defined color_type_intensity_gradient
		float w = getIntensity();
		color = texture2D(gradient, vec2(w,1.0-w)).rgb;
	#elif defined color_type_color
		color = uColor;
	#elif defined color_type_lod
		float depth = getLOD();
		float w = depth / 10.0;
		color = texture2D(gradient, vec2(w,1.0-w)).rgb;
	#elif defined color_type_point_index
		color = indices.rgb;
	#elif defined color_type_classification
		vec4 cl = getClassification(); 
		color = cl.rgb;
	#elif defined color_type_return_number
		color = getReturnNumber();
	#elif defined color_type_source
		color = getSourceID();
	#elif defined color_type_normal
		color = (modelMatrix * vec4(normal, 0.0)).xyz;
	#elif defined color_type_phong
		color = color;
	#elif defined color_type_composite
		color = getCompositeColor();
	#endif
	
    //if(false && (0.5 > snoise(vec4(position, 10.) / 100000000.)) ){ 
#ifdef color_type_rgb // debug mode : in color mode, apply density, else in elevation mode, apply density to color
    if(true && getNormalizedDistance() > rand_f(position.xyz)) { 
        gl_Position = vec4(100.0, 100.0, 100.0, 0.0); 
        color.r = 0.0; 
        color.g = 0.0; 
        color.b = 0.0; 
    }
#endif
	//color = vec3(downsamplingEllipseMatrix[0][0], downsamplingEllipseMatrix[0][1], downsamplingEllipseMatrix[0][2]); 
	
	//color = snoise(vec4(position, 10.)) * vec3(1.,1.,1.);
	
	/*
	vec3 position_bis = vec3(position.x, position.y, position.z); // position;
	//position_bis = position + 0.8 * getNormalizedDistance() * sin(uTime / 1.0) * normalize( normal ); 
	//position_bis.z = position_bis.z + 0.8 * 1. * sin(0.*getNormalizedDistance() - uTime / 1.0); //  getNormalizedDistance()
	//position_bis.z = position_bis.z + 0.5 * getNormalizedDistance() * sin(getNormalizedDistance() * 20. - uTime / 0.5); //  getNormalizedDistance()
	vec4 mvPosition = modelViewMatrix * vec4( position_bis, 1.0 ); 
	gl_Position = projectionMatrix * mvPosition; 
	
    // Add some perlin or simplex noise on top of the density rarefaction : see inigo quilez and thebookofshaders
    // https://gist.github.com/patriciogonzalezvivo/670c22f3966e662d2f83
    //color = getNormalizedDistance() * vec3(1.,1.,1.);
    if(true && (getNormalizedDistance() > rand_f(position.xyz)) ){ 
        vec3 dir = normalize(vec3(rand_f(position.xyz + uTime / 10000000.), rand_f(position.xyz+1.),rand_f(position.xyz+2.))) - .5;
        vec3 position_bis = position + 10. * dir; // getNormalizedDistance() 
        vec4 pt = vec4(position, uTime / 10.);
        position_bis = position + 2. * sin(uTime / 1.0) * getNormalizedDistance() * vec3(snoise(pt), snoise(pt+1.), snoise(pt+2.)) ;
        //position_bis = position_bis + (1. + pow(mod(uTime, 5.0), 3.)) * (position - vec3(modelMatrixInverse * vec4( downsamplingPoint, 1.0 )));

        vec4 mvPosition = modelViewMatrix * vec4( position_bis, 1.0 );
        gl_Position = 1. * projectionMatrix * mvPosition; 
		
	
    }
	*/
    // Black not facing vertices / Backface culling
    /*
    vec4 p = vec4( position, 1. );
    vec3 e = normalize( vec3( modelViewMatrix * p ) ); 
    vec3 n = get_normal(); 
    if (dot( n, e ) >0.) {
        color = color / 100.;
    }
    if(false || ((color.r <= 0.1) && (color.g >= 0.99) && (color.b <= 0.1))){ gl_Position = vec4(100.0, 100.0, 100.0, 0.0); }
    */
    

	return color;
}

float getPointSize(){
	float pointSize = 1.0;
	
	float slope = tan(fov / 2.0);
	float projFactor = -0.5 * uScreenHeight / (slope * vViewPosition.z);
	
	float r = uOctreeSpacing * 1.7;
	vRadius = r;
	#if defined fixed_point_size
		pointSize = size;
	#elif defined attenuated_point_size
		if(uUseOrthographicCamera){
			pointSize = size;
		}else{
			pointSize = size * spacing * projFactor;
			//pointSize = pointSize * projFactor;
		}
	#elif defined adaptive_point_size
		if(uUseOrthographicCamera) {
			float worldSpaceSize = 1.0 * size * r / getPointSizeAttenuation();
			pointSize = (worldSpaceSize / uOrthoWidth) * uScreenWidth;
		} else {

			if(uIsLeafNode && false){
				pointSize = size * spacing * projFactor;
			}else{
				float worldSpaceSize = 1.0 * size * r / getPointSizeAttenuation();
				pointSize = worldSpaceSize * projFactor;
			}
		}
	#endif

	pointSize = max(minSize, pointSize);
	pointSize = min(maxSize, pointSize);
	
	vRadius = pointSize / projFactor;

	return pointSize;
}

#if defined(num_clippolygons) && num_clippolygons > 0
bool pointInClipPolygon(vec3 point, int polyIdx) {

	mat4 wvp = uClipPolygonWVP[polyIdx];
	//vec4 screenClipPos = uClipPolygonVP[polyIdx] * modelMatrix * vec4(point, 1.0);
	//screenClipPos.xy = screenClipPos.xy / screenClipPos.w * 0.5 + 0.5;

	vec4 pointNDC = wvp * vec4(point, 1.0);
	pointNDC.xy = pointNDC.xy / pointNDC.w;

	int j = uClipPolygonVCount[polyIdx] - 1;
	bool c = false;
	for(int i = 0; i < 8; i++) {
		if(i == uClipPolygonVCount[polyIdx]) {
			break;
		}

		//vec4 verti = wvp * vec4(uClipPolygonVertices[polyIdx * 8 + i], 1);
		//vec4 vertj = wvp * vec4(uClipPolygonVertices[polyIdx * 8 + j], 1);

		//verti.xy = verti.xy / verti.w;
		//vertj.xy = vertj.xy / vertj.w;

		//verti.xy = verti.xy / verti.w * 0.5 + 0.5;
		//vertj.xy = vertj.xy / vertj.w * 0.5 + 0.5;

		vec3 verti = uClipPolygonVertices[polyIdx * 8 + i];
		vec3 vertj = uClipPolygonVertices[polyIdx * 8 + j];

		if( ((verti.y > pointNDC.y) != (vertj.y > pointNDC.y)) && 
			(pointNDC.x < (vertj.x-verti.x) * (pointNDC.y-verti.y) / (vertj.y-verti.y) + verti.x) ) {
			c = !c;
		}
		j = i;
	}

	return c;
}
#endif

void doClipping(){

	#if !defined color_type_composite
		vec4 cl = getClassification(); 
		if(cl.a == 0.0){
			gl_Position = vec4(100.0, 100.0, 100.0, 0.0);
			
			return;
		}
	#endif

	#if defined(clip_return_number_enabled)
	{ // return number filter
		vec2 range = uFilterReturnNumberRange;
		if(returnNumber < range.x || returnNumber > range.y){
			gl_Position = vec4(100.0, 100.0, 100.0, 0.0);
			
			return;
		}
	}
	#endif

	#if defined(clip_number_of_returns_enabled)
	{ // number of return filter
		vec2 range = uFilterNumberOfReturnsRange;
		if(numberOfReturns < range.x || numberOfReturns > range.y){
			gl_Position = vec4(100.0, 100.0, 100.0, 0.0);
			
			return;
		}
	}
	#endif

	#if defined(clip_gps_enabled)
	{ // GPS time filter
		float time = gpsTime + uGPSOffset;
		vec2 range = uFilterGPSTimeClipRange;

		if(time < range.x || time > range.y){
			gl_Position = vec4(100.0, 100.0, 100.0, 0.0);
			
			return;
		}
	}
	#endif

	int clipVolumesCount = 0;
	int insideCount = 0;

	#if defined(num_clipboxes) && num_clipboxes > 0
		for(int i = 0; i < num_clipboxes; i++){
			vec4 clipPosition = clipBoxes[i] * modelMatrix * vec4( position, 1.0 );
			bool inside = -0.5 <= clipPosition.x && clipPosition.x <= 0.5;
			inside = inside && -0.5 <= clipPosition.y && clipPosition.y <= 0.5;
			inside = inside && -0.5 <= clipPosition.z && clipPosition.z <= 0.5;

			insideCount = insideCount + (inside ? 1 : 0);
			clipVolumesCount++;
		}	
	#endif

	#if defined(num_clippolygons) && num_clippolygons > 0
		for(int i = 0; i < num_clippolygons; i++) {
			bool inside = pointInClipPolygon(position, i);

			insideCount = insideCount + (inside ? 1 : 0);
			clipVolumesCount++;
		}
	#endif

	bool insideAny = insideCount > 0;
	bool insideAll = (clipVolumesCount > 0) && (clipVolumesCount == insideCount);

	if(clipMethod == CLIPMETHOD_INSIDE_ANY){
		if(insideAny && clipTask == CLIPTASK_HIGHLIGHT){
			vColor.r += 0.5;
		}else if(!insideAny && clipTask == CLIPTASK_SHOW_INSIDE){
			gl_Position = vec4(100.0, 100.0, 100.0, 1.0);
		}else if(insideAny && clipTask == CLIPTASK_SHOW_OUTSIDE){
			gl_Position = vec4(100.0, 100.0, 100.0, 1.0);
		}
	}else if(clipMethod == CLIPMETHOD_INSIDE_ALL){
		if(insideAll && clipTask == CLIPTASK_HIGHLIGHT){
			vColor.r += 0.5;
		}else if(!insideAll && clipTask == CLIPTASK_SHOW_INSIDE){
			gl_Position = vec4(100.0, 100.0, 100.0, 1.0);
		}else if(insideAll && clipTask == CLIPTASK_SHOW_OUTSIDE){
			gl_Position = vec4(100.0, 100.0, 100.0, 1.0);
		}
	}
}



// 
// ##     ##    ###    #### ##    ## 
// ###   ###   ## ##    ##  ###   ## 
// #### ####  ##   ##   ##  ####  ## 
// ## ### ## ##     ##  ##  ## ## ## 
// ##     ## #########  ##  ##  #### 
// ##     ## ##     ##  ##  ##   ### 
// ##     ## ##     ## #### ##    ## 
//

void main() {
	vec4 mvPosition = modelViewMatrix * vec4( position, 1.0 );
	vViewPosition = mvPosition.xyz;
	gl_Position = projectionMatrix * mvPosition;
	vLogDepth = log2(-mvPosition.z);

	// POINT SIZE
	float pointSize = getPointSize();
	gl_PointSize = pointSize;
	vPointSize = pointSize;

	// COLOR
	vColor = getColor();


	#if defined hq_depth_pass
		float originalDepth = gl_Position.w;
		float adjustedDepth = originalDepth + 2.0 * vRadius;
		float adjust = adjustedDepth / originalDepth;

		mvPosition.xyz = mvPosition.xyz * adjust;
		gl_Position = projectionMatrix * mvPosition;
	#endif


	// CLIPPING
	doClipping();

	#if defined(num_clipspheres) && num_clipspheres > 0
		for(int i = 0; i < num_clipspheres; i++){
			vec4 sphereLocal = uClipSpheres[i] * mvPosition;

			float distance = length(sphereLocal.xyz);

			if(distance < 1.0){
				float w = distance;
				vec3 cGradient = texture2D(gradient, vec2(w, 1.0 - w)).rgb;
				
				//vColor = cGradient;
				//vColor = cGradient * 0.7 + vColor * 0.3;
			}
		}
	#endif

	#if defined(num_shadowmaps) && num_shadowmaps > 0

		const float sm_near = 0.1;
		const float sm_far = 10000.0;

		for(int i = 0; i < num_shadowmaps; i++){
			vec3 viewPos = (uShadowWorldView[i] * vec4(position, 1.0)).xyz;
			float distanceToLight = abs(viewPos.z);
			
			vec4 projPos = uShadowProj[i] * uShadowWorldView[i] * vec4(position, 1);
			vec3 nc = projPos.xyz / projPos.w;
			
			float u = nc.x * 0.5 + 0.5;
			float v = nc.y * 0.5 + 0.5;

			vec2 sampleStep = vec2(1.0 / (2.0*1024.0), 1.0 / (2.0*1024.0)) * 1.5;
			vec2 sampleLocations[9];
			sampleLocations[0] = vec2(0.0, 0.0);
			sampleLocations[1] = sampleStep;
			sampleLocations[2] = -sampleStep;
			sampleLocations[3] = vec2(sampleStep.x, -sampleStep.y);
			sampleLocations[4] = vec2(-sampleStep.x, sampleStep.y);

			sampleLocations[5] = vec2(0.0, sampleStep.y);
			sampleLocations[6] = vec2(0.0, -sampleStep.y);
			sampleLocations[7] = vec2(sampleStep.x, 0.0);
			sampleLocations[8] = vec2(-sampleStep.x, 0.0);

			float visibleSamples = 0.0;
			float numSamples = 0.0;

			float bias = vRadius * 2.0;

			for(int j = 0; j < 9; j++){
				vec4 depthMapValue = texture2D(uShadowMap[i], vec2(u, v) + sampleLocations[j]);

				float linearDepthFromSM = depthMapValue.x + bias;
				float linearDepthFromViewer = distanceToLight;

				if(linearDepthFromSM > linearDepthFromViewer){
					visibleSamples += 1.0;
				}

				numSamples += 1.0;
			}

			float visibility = visibleSamples / numSamples;

			if(u < 0.0 || u > 1.0 || v < 0.0 || v > 1.0 || nc.x < -1.0 || nc.x > 1.0 || nc.y < -1.0 || nc.y > 1.0 || nc.z < -1.0 || nc.z > 1.0){
				//vColor = vec3(0.0, 0.0, 0.2);
			}else{
				//vColor = vec3(1.0, 1.0, 1.0) * visibility + vec3(1.0, 1.0, 1.0) * vec3(0.5, 0.0, 0.0) * (1.0 - visibility);
				vColor = vColor * visibility + vColor * uShadowColor * (1.0 - visibility);
			}
		}

	#endif

	//vColor = vec3(1.0, 0.0, 0.0);

	//if(uDebug){
	//	vColor.b = (vColor.r + vColor.g + vColor.b) / 3.0;
	//	vColor.r = 1.0;
	//	vColor.g = 1.0;
	//}

}
