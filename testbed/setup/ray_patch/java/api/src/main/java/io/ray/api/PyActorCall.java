// Generated by `RayCallGenerator.java`. DO NOT EDIT.

package io.ray.api;

import io.ray.api.call.PyActorTaskCaller;
import io.ray.api.function.PyActorMethod;

/**
 * This class provides type-safe interfaces for remote actor calls.
 **/
interface PyActorCall {

  default <R> PyActorTaskCaller<R> task(PyActorMethod<R> pyActorMethod) {
    Object[] args = new Object[]{};
    return new PyActorTaskCaller<>((PyActorHandle)this, pyActorMethod, args);
  }

  default <R> PyActorTaskCaller<R> task(PyActorMethod<R> pyActorMethod, Object obj0) {
    Object[] args = new Object[]{obj0};
    return new PyActorTaskCaller<>((PyActorHandle)this, pyActorMethod, args);
  }

  default <R> PyActorTaskCaller<R> task(PyActorMethod<R> pyActorMethod, Object obj0, Object obj1) {
    Object[] args = new Object[]{obj0, obj1};
    return new PyActorTaskCaller<>((PyActorHandle)this, pyActorMethod, args);
  }

  default <R> PyActorTaskCaller<R> task(PyActorMethod<R> pyActorMethod, Object obj0, Object obj1, Object obj2) {
    Object[] args = new Object[]{obj0, obj1, obj2};
    return new PyActorTaskCaller<>((PyActorHandle)this, pyActorMethod, args);
  }

  default <R> PyActorTaskCaller<R> task(PyActorMethod<R> pyActorMethod, Object obj0, Object obj1, Object obj2, Object obj3) {
    Object[] args = new Object[]{obj0, obj1, obj2, obj3};
    return new PyActorTaskCaller<>((PyActorHandle)this, pyActorMethod, args);
  }

  default <R> PyActorTaskCaller<R> task(PyActorMethod<R> pyActorMethod, Object obj0, Object obj1, Object obj2, Object obj3, Object obj4) {
    Object[] args = new Object[]{obj0, obj1, obj2, obj3, obj4};
    return new PyActorTaskCaller<>((PyActorHandle)this, pyActorMethod, args);
  }

}