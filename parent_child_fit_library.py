# Replica of several functions from model.py in the hyperspy package.
# https://github.com/hyperspy/hyperspy/blob/RELEASE_next_minor/hyperspy/model.py
# Author: Sanket Gadgil, Date: 16/11/2020

from hyperspy.misc.utils import dummy_context_manager
from hyperspy.docstrings.signal import SHOW_PROGRESSBAR_ARG
from hyperspy.external.progressbar import progressbar
from hyperspy.external.mpfit.mpfit import mpfit
from scipy.optimize import (leastsq, OptimizeResult)
import numpy as np
import logging


_logger = logging.getLogger(__name__)


def fit(
    model,
    old_p1,
    grad="fd",
    update_plot=False,
    return_info=True,
    **kwargs,
):
    '''
    Replica of fit() function from hyperspy/model.py, massively simplified,
    with multiple assumed variable values.

    Arguments:
    model -- The model to be fitted, this needs to be passed as fit(),
             in this case, is not a member function of the Model class
             as it is in model.py
    old_p1 -- Inherited parameter set.
    grad -- Gradient mode, default value of "fd" meaning finite difference
    update_plot -- Flag indicating whether to update any potential interactive
                   plot
    return_info -- Flag indicating whether or not to return results from
                   function
    **kwargs -- Keyword arguments to pass to leastsq()

    Output:
    model.fit_output -- If return_info=True, return the output of the
                        fit function, leastsq()
    '''

    # Context manager, not quite sure what this does
    cm = (
        model.suspend_update
        if (update_plot != model._plot_active) and not update_plot
        else dummy_context_manager
    )

    with cm(update_on_resume=True):
        model.p_std = None
        # model._set_p0()  # Original version of setting model.p0
        model.p0 = old_p1  # Set existing p0 to inherited p0
        old_p0 = model.p0  # For checking fit function success later

        weights = model._convert_variance_to_weights()

        args = (model.signal()[np.where(model.channel_switches)], weights)

        grad = None

        # Actual optimization function using scipy.optimize.leastsq()
        # The model._errfunc calls a function which is defined in
        # ScalableReferencePattern.function() in pyxem/components/scalable_reference_pattern.py
        res = leastsq(
            model._errfunc,
            model.p0[:],
            Dfun=grad,
            col_deriv=1,
            args=args,
            full_output=True,
            **kwargs,
        )

        # Create an object, OptimizeResult, to store optimization results
        model.fit_output = OptimizeResult(
            x=res[0],
            covar=res[1],
            fun=res[2]["fvec"],
            nfev=res[2]["nfev"],
            success=res[4] in [1, 2, 3, 4],
            status=res[4],
            message=res[3],
        )

        model.p0 = model.fit_output.x  # Update parameter set in model
        ysize = len(model.fit_output.fun)
        cost = np.sum(model.fit_output.fun ** 2)
        pcov = model.fit_output.covar  # Covariance of parameter fit results

        # Standard deviation of parameter fit results
        model.p_std = model._calculate_parameter_std(pcov, cost, ysize)

        # If only 1 parameter is in set p0 then turn into a list of
        # parameters 1 element long (for the sake of compatibility)
        if np.iterable(model.p0) == 0:
            model.p0 = (model.p0,)

        # Store parameter values in a map corresponding to the image data used
        # to create the model
        model._fetch_values_from_p0(p_std=model.p_std)
        model.store_current_values()
        model._calculate_chisq()
        model._set_current_degrees_of_freedom()

    # Check that the parameters have been changed by the fitting
    if np.any(old_p0 != model.p0):
        model.events.fitted.trigger(model)

    # Success check and error message output
    success = model.fit_output.get("success", None)
    if success is False:
        message = model.fit_output.get("message", "Unknown reason")
        _logger.warning(
            f"`m.fit()` did not exit successfully. Reason: {message}"
        )

    if return_info:
        return model.fit_output
    else:
        return None


def boundedfit(
    model,
    old_p1,
    bounded=True,
    update_plot=False,
    return_info=True,
    **kwargs,
):
    '''
    Replica of bounded fit function from hyperspy/model.py, massively
    simplified, with multiple assumed variable values.

    Arguments:
    model -- The model to be fitted, this needs to be passed as fit(),
             in this case, is not a member function of the Model class
             as it is in model.py
    old_p1 -- Inherited parameter set.
    bounded -- Flag indicating whether bounded fit function should be used.
               Not strictly necessary for this function but a remnant from
               the hyperspy implementation where bounded fit is enclosed as
               an option inside fit() in hyperspy/model.py.
    update_plot -- Flag indicating whether to update any potential interactive
                   plot
    return_info -- Flag indicating whether or not to return results from
                   function
    **kwargs -- Keyword arguments to pass to mpfit() from
                hyperspy.external.mpfit.mpfit.

    Output:
    model.fit_output -- If return_info=True, return the output of the
                        fit function, mpfit()
    '''

    # Context manager, not quite sure what this does
    cm = (
        model.suspend_update
        if (update_plot != model._plot_active) and not update_plot
        else dummy_context_manager
    )

    # Bind existing parameters inside their prescribed limits
    if bounded:
        model.ensure_parameters_in_bounds()

    with cm(update_on_resume=True):
        model.p_std = None
        model.p0 = old_p1  # Set existing p0 to inherited p0
        old_p0 = model.p0  # For checking fit function success later

        weights = model._convert_variance_to_weights()

        args = (model.signal()[np.where(model.channel_switches)], weights)

        model._set_mpfit_parameters_info(bounded=bounded)

        auto_deriv = 1

        # Actual optimization function using hyperspy.external.mpfit.mpfit
        # The model._errfunc4mpfit calls a function which is defined in
        # ScalableReferencePattern.function() in pyxem/components/scalable_reference_pattern.py
        # model.mpfit_parinfo contains information about the limits set for the parameters
        res = mpfit(
            model._errfunc4mpfit,
            model.p0[:],
            parinfo=model.mpfit_parinfo,
            functkw={
                "y": model.signal()[model.channel_switches],
                "weights": weights,
            },
            autoderivative=auto_deriv,
            quiet=1,
            **kwargs,
        )

        # Create an object, OptimizeResult, to store optimization results
        model.fit_output = OptimizeResult(
            x=res.params,
            covar=res.covar,
            perror=res.perror,
            nit=res.niter,
            nfev=res.nfev,
            success=(res.status > 0) and (res.status != 5),
            status=res.status,
            message=res.errmsg,
            debug=res.debug,
            dof=res.dof,
            fnorm=res.fnorm,
        )

        model.p0 = model.fit_output.x  # Update parameter set in model
        ysize = len(model.fit_output.x) + model.fit_output.dof
        cost = model.fit_output.fnorm
        pcov = model.fit_output.perror ** 2  # Covariance of parameter fit results

        # Standard deviation of parameter fit results
        model.p_std = model._calculate_parameter_std(pcov, cost, ysize)

        # If only 1 parameter is in set p0 then turn into a list of
        # parameters 1 element long (for the sake of compatibility)
        if np.iterable(model.p0) == 0:
            model.p0 = (model.p0,)

        # Store parameter values in a map corresponding to the image data used
        # to create the model
        model._fetch_values_from_p0(p_std=model.p_std)
        model.store_current_values()
        model._calculate_chisq()
        model._set_current_degrees_of_freedom()

    # Check that the parameters have been changed by the fitting
    if np.any(old_p0 != model.p0):
        model.events.fitted.trigger(model)

    # Success check and error message output
    success = model.fit_output.get("success", None)
    if success is False:
        message = model.fit_output.get("message", "Unknown reason")
        _logger.warning(
            f"`m.fit()` did not exit successfully. Reason: {message}"
        )

    if return_info:
        return model.fit_output
    else:
        return None


def multifit(
    model,
    firstfit=False,
    bounded=False,
    fetch_only_fixed=False,
    show_progressbar=True,
    iterpath=None,
    **kwargs,
):
    '''
    Replica of multi-dimensional fit function from hyperspy/model.py, massively
    simplified, with multiple assumed variable values.

    Arguments:
    model -- The model to be fitted, this needs to be passed as fit(),
             in this case, is not a member function of the Model class
             as it is in model.py
    firstfit -- Flag indicating whether the call to this function pertains to the
                first level of the parent-child algorithm.
    bounded -- Flag indicating whether the fit performed should be bounded or not.
    fetch_only_fixed -- Flag indicating whether to fetch only fixed parameters.
    show_progressbar -- Flag indicating whether to show a progress bar or not.
    iterpath -- Flag indicating the path that the iteration through the 2D image
                should take, either "flyback" or "serpentine".

    Output:
    None
    '''

    # Setup progress bar and iteration path
    maxval = model.axes_manager.navigation_size
    show_progressbar = show_progressbar and (maxval > 0)
    model.axes_manager._iterpath = iterpath

    NavAxesSize = model.axes_manager.navigation_axes[0].size

    # Initialize and set inherited parameters.
    inherited_params = np.zeros(
        (NavAxesSize, NavAxesSize, len(model[0].free_parameters))
    )
    for index in model.axes_manager:
        for count, param in enumerate(model[0].free_parameters):
            inherited_params[index][count] = param.map["values"][index]

    # Main loop
    i = 0
    with model.axes_manager.events.indices_changed.suppress_callback(
        model.fetch_stored_values
    ):
        # Unclear what these do as they relate to context managers.
        outer = model.suspend_update
        inner = dummy_context_manager

        with outer(update_on_resume=True):
            with progressbar(
                total=maxval, disable=not show_progressbar, leave=True
            ) as pbar:
                # Original parameters: (1, 0, 0, 1, 0 ,0). To be used if the function is called
                # for the first level of the parent-child algorithm.
                orig_params = \
                    tuple([param.value for param in model[0].free_parameters])
                for index in model.axes_manager:  # iterate through the pixels in the 2D image

                    with inner(update_on_resume=True):

                        model.fetch_stored_values(only_fixed=fetch_only_fixed)

                        # Conditional calls to either fit() or boundedfit()
                        if bounded:
                            if firstfit:
                                fit_results = boundedfit(
                                        model,
                                        old_p1=orig_params,
                                        **kwargs
                                    )
                            else:
                                fit_results = boundedfit(
                                        model,
                                        old_p1=inherited_params[index],
                                        **kwargs
                                    )
                        else:
                            if firstfit:
                                fit_results = fit(
                                    model,
                                    old_p1=orig_params,
                                    **kwargs
                                )
                            else:
                                fit_results = fit(
                                    model,
                                    old_p1=inherited_params[index],
                                    **kwargs
                                )

                        # Update the progress bar.
                        i += 1
                        pbar.update(1)
