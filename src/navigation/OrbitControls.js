/**
 * @author mschuetz / http://mschuetz.at
 *
 * adapted from THREE.OrbitControls by
 *
 * @author qiao / https://github.com/qiao
 * @author mrdoob / http://mrdoob.com
 * @author alteredq / http://alteredqualia.com/
 * @author WestLangley / http://github.com/WestLangley
 * @author erich666 / http://erichaines.com
 *
 *
 *
 */


import {MOUSE} from "../defines.js";
import {Utils} from "../utils.js";
import {EventDispatcher} from "../EventDispatcher.js";

 
export class OrbitControls extends EventDispatcher{
	
	constructor(viewer){
		super();
		
		this.viewer = viewer;
		this.renderer = viewer.renderer;

		this.scene = null;
		this.sceneControls = new THREE.Scene();

		this.rotationSpeed = 5;
        this.panSpeed = 1.; 

		this.fadeFactor = 10;
		this.yawDelta = 0;
		this.pitchDelta = 0;
		this.panDelta = new THREE.Vector2(0, 0);
		this.radiusDelta = 0;

		this.tweens = [];

		let drag = (e) => {
			if (e.drag.object !== null) {
				return;
			}

			if (e.drag.startHandled === undefined) {
				e.drag.startHandled = true;

				this.dispatchEvent({type: 'start'});
			}

			let ndrag = {
				x: e.drag.lastDrag.x / this.renderer.domElement.clientWidth,
				y: e.drag.lastDrag.y / this.renderer.domElement.clientHeight
			};

			if (e.drag.mouse === MOUSE.LEFT) {
				this.yawDelta += ndrag.x * this.rotationSpeed;
				this.pitchDelta += ndrag.y * this.rotationSpeed;

				this.stopTweens();
			} else if (e.drag.mouse === MOUSE.RIGHT) {
				this.panDelta.x += ndrag.x;
				this.panDelta.y += ndrag.y;

				this.stopTweens();
			}
		};

		let drop = e => {
			this.dispatchEvent({type: 'end'});
		};

		let scroll = (e) => {
			let resolvedRadius = this.scene.view.radius + this.radiusDelta;

			this.radiusDelta += -e.delta * resolvedRadius * 0.1;

			this.stopTweens();
		};

		let dblclick = (e) => {
			this.zoomToLocation(e.mouse);
		};

		let previousTouch = null;
		let touchStart = e => {
			previousTouch = e;
		};

		let touchEnd = e => {
			previousTouch = e;
		};

		let touchMove = e => {
			if (e.touches.length === 2 && previousTouch.touches.length === 2){
				let prev = previousTouch;
				let curr = e;

				let prevDX = prev.touches[0].pageX - prev.touches[1].pageX;
				let prevDY = prev.touches[0].pageY - prev.touches[1].pageY;
				let prevDist = Math.sqrt(prevDX * prevDX + prevDY * prevDY);

				let currDX = curr.touches[0].pageX - curr.touches[1].pageX;
				let currDY = curr.touches[0].pageY - curr.touches[1].pageY;
				let currDist = Math.sqrt(currDX * currDX + currDY * currDY);

				let delta = currDist / prevDist;
				let resolvedRadius = this.scene.view.radius + this.radiusDelta;
				let newRadius = resolvedRadius / delta;
				this.radiusDelta = newRadius - resolvedRadius;

				this.stopTweens();
			}else if(e.touches.length === 3 && previousTouch.touches.length === 3){
				let prev = previousTouch;
				let curr = e;

				let prevMeanX = (prev.touches[0].pageX + prev.touches[1].pageX + prev.touches[2].pageX) / 3;
				let prevMeanY = (prev.touches[0].pageY + prev.touches[1].pageY + prev.touches[2].pageY) / 3;

				let currMeanX = (curr.touches[0].pageX + curr.touches[1].pageX + curr.touches[2].pageX) / 3;
				let currMeanY = (curr.touches[0].pageY + curr.touches[1].pageY + curr.touches[2].pageY) / 3;

				let delta = {
					x: (currMeanX - prevMeanX) / this.renderer.domElement.clientWidth,
					y: (currMeanY - prevMeanY) / this.renderer.domElement.clientHeight
				};

				// this.panDelta.x += delta.x;
				// this.panDelta.y += delta.y;
				this.panDelta.x = delta.x * this.panSpeed;
				this.panDelta.y = delta.y * this.panSpeed;

				this.stopTweens();
			}

			previousTouch = e;
		};

		this.addEventListener('touchstart', touchStart);
		this.addEventListener('touchend', touchEnd);
		this.addEventListener('touchmove', touchMove);
		this.addEventListener('drag', drag);
		this.addEventListener('drop', drop);
		this.addEventListener('mousewheel', scroll);
		this.addEventListener('dblclick', dblclick);
        
        // controls_limits to clamp to to avoid weird relocation in toucscreen controls
        // Slower pan
		this.touchscreenMode = true;
        if (this.touchscreenMode) {
            this.rotationSpeed = 2;
            this.panSpeed = 1. / 10.; 
        }
        // controls example for bamiyan cliff
        this.controls_limits = {
            cam_pos: {
                min: new THREE.Vector3(391600, 3852400, -13), 
                max: new THREE.Vector3(393200, 3857300, 1500)
            }, 
            pivot: {
                min: new THREE.Vector3(391600, 3854400, -100), 
                max: new THREE.Vector3(393200, 3855300, 170)
            },
            yaw: {
                min: -1.5, 
                max: 1.5
            }, 
            pitch: {
                min: -10, 
                max: 0.5
            }, 
            radius: {
                min: 5, 
                max: 1500
            }
        };
	}

	setScene (scene) {
		this.scene = scene;
	}

	stop(){
		this.yawDelta = 0;
		this.pitchDelta = 0;
		this.radiusDelta = 0;
		this.panDelta.set(0, 0);
	}
	
	zoomToLocation(mouse){
		let camera = this.scene.getActiveCamera();
		
		let I = Utils.getMousePointCloudIntersection(
			mouse,
			camera,
			this.viewer,
			this.scene.pointclouds,
			{pickClipped: true});

		if (I === null) {
			return;
		}

		let targetRadius = 0;
		{
			let minimumJumpDistance = 0.2;

			let domElement = this.renderer.domElement;
			let ray = Utils.mouseToRay(mouse, camera, domElement.clientWidth, domElement.clientHeight);

			let nodes = I.pointcloud.nodesOnRay(I.pointcloud.visibleNodes, ray);
			let lastNode = nodes[nodes.length - 1];
			let radius = lastNode.getBoundingSphere(new THREE.Sphere()).radius;
			targetRadius = Math.min(this.scene.view.radius, radius);
			targetRadius = Math.max(minimumJumpDistance, targetRadius);
		}

		let d = this.scene.view.direction.multiplyScalar(-1);
		let cameraTargetPosition = new THREE.Vector3().addVectors(I.location, d.multiplyScalar(targetRadius));
		// TODO Unused: let controlsTargetPosition = I.location;

		let animationDuration = 600;
		let easing = TWEEN.Easing.Quartic.Out;

		{ // animate
			let value = {x: 0};
			let tween = new TWEEN.Tween(value).to({x: 1}, animationDuration);
			tween.easing(easing);
			this.tweens.push(tween);

			let startPos = this.scene.view.position.clone();
			let targetPos = cameraTargetPosition.clone();
			let startRadius = this.scene.view.radius;
			let targetRadius = cameraTargetPosition.distanceTo(I.location);

			tween.onUpdate(() => {
				let t = value.x;
				this.scene.view.position.x = (1 - t) * startPos.x + t * targetPos.x;
				this.scene.view.position.y = (1 - t) * startPos.y + t * targetPos.y;
				this.scene.view.position.z = (1 - t) * startPos.z + t * targetPos.z;

				this.scene.view.radius = (1 - t) * startRadius + t * targetRadius;
				this.viewer.setMoveSpeed(this.scene.view.radius / 2.5);
			});

			tween.onComplete(() => {
				this.tweens = this.tweens.filter(e => e !== tween);
			});

			tween.start();
		}
	}

	stopTweens () {
		this.tweens.forEach(e => e.stop());
		this.tweens = [];
	}

	update (delta) {
		let view = this.scene.view;

		{ // apply rotation
			let progression = Math.min(1, this.fadeFactor * delta);

			let yaw = view.yaw;
			let pitch = view.pitch;
			let pivot = view.getPivot();

			yaw -= progression * this.yawDelta;
			pitch -= progression * this.pitchDelta;

			// limit pitch and yaw, and eventually restore
			let prevYaw = view.yaw;
			let prevPitch = view.pitch;
			yaw = Math.max(controls_limits.yaw.min, Math.min(controls_limits.yaw.max, yaw));
			pitch = Math.max(controls_limits.pitch.min, Math.min(controls_limits.pitch.max, pitch));

			view.yaw = yaw;
			view.pitch = pitch;

			let V = this.scene.view.direction.multiplyScalar(-view.radius);
			let position = new THREE.Vector3().addVectors(pivot, V);

			// Avoid weird pivot modification if cam going outside of bounds
			if (position.equals(position.clone().clamp(controls_limits.cam_pos.min, controls_limits.cam_pos.max))) {
				view.position.copy(position);
			} else {
				view.yaw = prevYaw;
				view.pitch = prevPitch;
			}
		}

		{ // apply pan
			let progression = Math.min(1, this.fadeFactor * delta);
			let panDistance = progression * view.radius * 3;

			let px = -this.panDelta.x * panDistance;
			let py = this.panDelta.y * panDistance;

			view.pan(px, py);
            
			// Limit pan to keep pivot/target in AABB
			let pivot_in_box = viewer.scene.view.getPivot().clone();
			pivot_in_box.clamp(controls_limits.pivot.min, controls_limits.pivot.max);
			view.position = new THREE.Vector3().addVectors(pivot_in_box, view.direction.multiplyScalar(-view.radius));
			// clamp position inside box
			view.position.clamp(controls_limits.cam_pos.min, controls_limits.cam_pos.max);
		}

		{ // apply zoom
			let progression = Math.min(1, this.fadeFactor * delta);

			// let radius = view.radius + progression * this.radiusDelta * view.radius * 0.1;
			let radius = view.radius + progression * this.radiusDelta;

			// limit radius
			radius = Math.max(controls_limits.radius.min, Math.min(controls_limits.radius.max, radius));
		
			let V = view.direction.multiplyScalar(-radius);
			let position = new THREE.Vector3().addVectors(view.getPivot(), V);
			view.radius = radius;

			view.position.copy(position);
		}

		{
			let speed = view.radius / 2.5;
			this.viewer.setMoveSpeed(speed);
		}

		{ // decelerate over time
			let progression = Math.min(1, this.fadeFactor * delta);
			let attenuation = Math.max(0, 1 - this.fadeFactor * delta);

			this.yawDelta *= attenuation;
			this.pitchDelta *= attenuation;
			this.panDelta.multiplyScalar(attenuation);
			// this.radiusDelta *= attenuation;
			this.radiusDelta -= progression * this.radiusDelta;
		}
        
		// UPDATE PIVOT ONCE MOTION DONE, to avoid going behind geometry
		/*
		let camera = this.scene.getActiveCamera();
		camera.position.copy(this.scene.view.position);
		camera.rotation.order = "ZXY";
		camera.rotation.x = Math.PI / 2 + this.scene.view.pitch;
		camera.rotation.z = this.scene.view.yaw;
		camera.updateMatrix();
		camera.updateMatrixWorld();
		camera.matrixWorldInverse.getInverse(camera.matrixWorld);
		let I = Potree.utils.getMousePointCloudIntersection(
			new THREE.Vector2(this.renderer.domElement.clientWidth / 2, this.renderer.domElement.clientHeight / 2),
			camera,
			this.viewer,
			this.scene.pointclouds,
			{pickClipped: true});
		if (I === null) {
			return;
		} else {
			let new_pivot = I.location;
			//console.log(new_pivot);
			view.radius = view.position.distanceTo(new_pivot);
			view.position = new THREE.Vector3().addVectors(new_pivot, view.direction.multiplyScalar(-view.radius));
		}
		*/
	}
};
