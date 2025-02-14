struct Callback{C,E}
    "condition for performing callback: (iter, state, H, envs) -> bool"
    condition::C
    "callback function acting on solver state: (iter, state, H, envs) -> state, envs"
    affect!!::E
end

function (c::Callback)(iter, state, H, envs)
    if c.condition(iter, state, H, envs)
        state, envs = c.affect!!(iter, state, H, envs)
    end

    return state, envs
end

struct CallbackList <: Callback
    callbacks::Vector{Callback}
end

function (c::CallbackList)(iter, state, H, envs)
    for callback in c.callbacks
        state, envs = callback(iter, state, H, envs)
    end

    return state, envs
end

Base.getindex(c::CallbackList, idx) = getindex(c.callbacks, idx)
Base.length(c::CallbackList) = length(c.callbacks)

############################################
# Record OBSERVABLES DURING OPTIMIZATION (EFFECT)
############################################
struct RecordObservable{T1,T2}
    "collection of string-array pairs containing observable data"
    data::T1
    "functions to compute observable data"
    observables::T2

    "expects iterable of key-value pairs (observable_name, function_to_compute_observable)"
    function RecordObservable(recipe)
        names = keys(recipe)
        observables = values(recipe)
        data = NamedTuple{names}(([] for _ in eachindex(names)))
        return new{typeof(data),typeof(observables)}(data, observables)
    end

    function RecordObservable(data, observables)
        return new{typeof(data),typeof(observables)}(data, observables)
    end
end

Base.length(p::RecordObservable) = length(p.data)
Base.iterate(p::RecordObservable) = iterate(p.data)
Base.iterate(p::RecordObservable, state) = iterate(p.data, state)

function Base.:&(p1::RecordObservable, p2::RecordObservable)
    return RecordObservable(merge(p1.data, p2.data), [p1.observables, p2.observables])
end

function (p::RecordObservable)(iter, state, H, envs)
    for i in 1:length(p)
        push!(p.data[i], p.observables[i](iter, state, H, envs))
    end

    return state, envs
end

############################################
# WAVE-FUNCTION BACKUP DURING OPTIMIZATION (EFFECT)
############################################

struct SaveState
    "system parameters and state"
    data::Dict
    "optional observables to save"
    observables::RecordObservable
    "toggle to save checkpoints to file every tick"
    save_every_tick::Bool
    "path to save checkpoint files"
    savepath::String

    function SaveState(params::Dict, observables::Union{Nothing,RecordObservable}, path,
                       save_every_tick=true)
        isnothing(observables) && (observables = RecordObservable(()))
        obj = new(merge(params, Dict("state" => Nothing, "envs" => Nothing)), observables,
                  nothing, save_every_tick, path)

        # save when program is interrupted
        atexit(() -> save(obj))

        return obj
    end
end

function SaveState(params, observable_callback::Callback{<:Any,<:RecordObservable}, path,
                   save_every_tick=true)
    return SaveState(params, observable_callback.affect!!, path, save_every_tick)
end

function (p::SaveState)(iter, state, H, envs)
    p.psi = copy(state)
    p.envs = copy(envs)

    p.save_every_tick && save(p)
    return state, envs
end

function save(p::SaveState)
    if isnothing(p.data["state"])
        @warn "Program terminated before a checkpoint was reached. State was not saved!"
        save(p.savepath, "data", p.data)
    end
end

############################################
# COMMON CALLBACK EFFECTS
############################################

function RecordEnergyConvergence()
    return RecordObservable((energies=(iter, state, H, envs) -> real(expectation_value(state,
                                                                                       H,
                                                                                       envs)),
                             times=(iter, state, H, envs) -> Base.time(),
                             errors=(iter, state, H, envs) -> maximum(pos -> calc_galerkin(state,
                                                                                           pos,
                                                                                           envs),
                                                                      1:length(state))))
end

############################################
# COMMON CALLBACK CONDITIONS
############################################
abstract type CallbackCondition end

struct OnAny{T<:Tuple{Vararg{CallbackCondition}}} <: CallbackCondition
    conditions::T
end

function (p::OnAny)(iter, state, H, envs)
    return any(condition -> condition(iter, state, H, envs) for condition in p.conditions)
end

Base.:|(p1::CallbackCondition, p2::CallbackCondition) = OnAny((p1, p2))
Base.:|(p1::OnAny, p2::CallbackCondition) = OnAny((p1.conditions..., p2))
Base.:|(p1::CallbackCondition, p2::OnAny) = OnAny((p1, p2.conditions...))
Base.:|(p1::OnAny, p2::OnAny) = OnAny((p1.conditions..., p2.conditions...))

struct OnAll{T<:Tuple{Vararg{CallbackCondition}}} <: CallbackCondition
    conditions::T
end

function (p::OnAll)(iter, state, H, envs)
    return all(condition -> condition(iter, state, H, envs) for condition in p.conditions)
end

Base.:&(p1::CallbackCondition, p2::CallbackCondition) = OnAll((p1, p2))
Base.:&(p1::OnAll, p2::CallbackCondition) = OnAll((p1.conditions..., p2))
Base.:&(p1::CallbackCondition, p2::OnAll) = OnAll((p1, p2.conditions...))
Base.:&(p1::OnAll, p2::OnAll) = OnAll((p1.conditions..., p2.conditions...))

struct OnIterElapsed <: CallbackCondition
    "number of iterations between trigger"
    save_freq::Int
end

(p::OnIterElapsed)(iter, state, H, envs) = iszero(iter % p.save_freq)

mutable struct OnTimeElapsed <: CallbackCondition
    "starting time in seconds"
    start_tick::Float64
    "number of seconds between trigger"
    save_freq::Float64

    # :s - second, :m - minute, :h - hour
    function OnTimeElapsed(freq, unit=:m)
        if unit == :m
            freq *= 60
        elseif unit == :h
            freq *= 3600
        elseif unit != :s
            throw(ArgumentError("invalid unit `:$unit`, expected `:s`, `:m` or `:h`"))
        end

        return new(time(), freq)
    end
end

# only gets called AFTER an iteration is complete!
function (p::OnTimeElapsed)(iter, state, H, envs)
    return ((time() - p.start_tick) > p.save_freq) ? (p.start_tick = time(); true) : false
end
