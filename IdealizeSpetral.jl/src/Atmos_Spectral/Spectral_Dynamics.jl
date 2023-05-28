export Compute_Corrections_Init, Compute_Corrections!, Four_In_One!, Spectral_Dynamics!, Get_Topography!, Spectral_Initialize_Fields!, Spectral_Dynamics_Physics!, Atmosphere_Update!



function Compute_Corrections_Init(vert_coord::Vert_Coordinate, mesh::Spectral_Spherical_Mesh, atmo_data::Atmo_Data,
    grid_u_p::Array{Float64, 3}, grid_v_p::Array{Float64, 3}, grid_ps_p::Array{Float64, 3}, grid_t_p::Array{Float64, 3}, 
    grid_δu::Array{Float64, 3}, grid_δv::Array{Float64, 3}, grid_δt::Array{Float64, 3},  
    Δt::Int64, grid_energy_temp::Array{Float64, 3}, grid_tracers_p::Array{Float64, 3}, grid_tracers_c::Array{Float64, 3}, grid_δtracers::Array{Float64,3}, grid_tracers_all::Array{Float64,3})
    
    do_mass_correction, do_energy_correction, do_water_correction = atmo_data.do_mass_correction, atmo_data.do_energy_correction, atmo_data.do_water_correction
    
    sum_tracers_p = 0.

    if (do_mass_correction) 
        mean_ps_p = Area_Weighted_Global_Mean(mesh, grid_ps_p)
    end
    
    if (do_energy_correction) 
        # due to dissipation introduced by the forcing
        cp_air, grav = atmo_data.cp_air, atmo_data.grav
        grid_energy_temp  .=  0.5*((grid_u_p + Δt*grid_δu).^2 + (grid_v_p + Δt*grid_δv).^2) + cp_air*(grid_t_p + Δt*grid_δt)
        mean_energy_p = Mass_Weighted_Global_Integral(vert_coord, mesh, atmo_data, grid_energy_temp, grid_ps_p)
    end
    
    if (do_water_correction)
        # By CJY 0517
        grid_tracers_p[grid_tracers_p .< 0] .= 0
        grid_tracers_c[grid_tracers_c .< 0] .= 0
        grid_tracers_all .= grid_tracers_p .+ Δt*grid_δtracers
        grid_tracers_all[grid_tracers_all .< 0] .= 0
        sum_tracers_p = qv_Global_Integral_to_e(vert_coord, mesh, atmo_data, grid_tracers_all, grid_ps_p)

        @info "sum_tracer_p" sum_tracers_p ./ (4. * pi * atmo_data.radius^2)
        # sum_tracers_p    = grav .* Mass_Weighted_Global_Integral(vert_coord, mesh, atmo_data, grid_tracers_all, grid_ps_p) .* 128. .* 64. .* 20.
    end
    
    return mean_ps_p, mean_energy_p, sum_tracers_p
end 

function Compute_Corrections!(vert_coord::Vert_Coordinate, mesh::Spectral_Spherical_Mesh, atmo_data::Atmo_Data,
    mean_ps_p::Float64, mean_energy_p::Float64, 
    grid_u_n::Array{Float64, 3}, grid_v_n::Array{Float64, 3},
    grid_energy_temp::Array{Float64, 3},
    grid_ps_n::Array{Float64, 3}, spe_lnps_n::Array{ComplexF64, 3}, 
    grid_t_n::Array{Float64, 3}, spe_t_n::Array{ComplexF64, 3},
    sum_tracers_p::Float64, grid_tracers_p::Array{Float64, 3}, grid_tracers_c::Array{Float64, 3}, grid_tracers_n::Array{Float64, 3}, 
    grid_t::Array{Float64, 3}, 
    grid_p_full::Array{Float64, 3})


    do_mass_correction, do_energy_correction, do_water_correction = atmo_data.do_mass_correction, atmo_data.do_energy_correction, atmo_data.do_water_correction
    
    
    if (do_mass_correction) 
        mean_ps_n = Area_Weighted_Global_Mean(mesh, grid_ps_n)
        mass_correction_factor = mean_ps_p/mean_ps_n
        grid_ps_n .*= mass_correction_factor
        #P00 = 1 
        spe_lnps_n[1,1,1] += log(mass_correction_factor)
    end
    
    if (do_energy_correction) 
        cp_air, grav = atmo_data.cp_air, atmo_data.grav
        
        grid_energy_temp .=  0.5*(grid_u_n.^2 + grid_v_n.^2) + cp_air*grid_t_n
        mean_energy_n = Mass_Weighted_Global_Integral(vert_coord, mesh, atmo_data, grid_energy_temp, grid_ps_n)
        
        temperature_correction = grav*(mean_energy_p - mean_energy_n)/(cp_air*mean_ps_p)
        #@info grav, mean_energy_p , mean_energy_n, cp_air, mean_ps_p
        grid_t_n .+= temperature_correction
        spe_t_n[1,1,:] .+= temperature_correction

    end

    # @info mean_ps_p, mean_energy_p, mass_correction_factor, temperature_correction
    # error(6868)
    ### By CJY 0517
    nλ = mesh.nλ
    nθ = mesh.nθ
    nd = mesh.nd
    grav = atmo_data.grav
    if (do_water_correction) 
        """
        grid_tracers_p[grid_tracers_p .< 0] .= 0
        grid_tracers_n[grid_tracers_n .< 0] .= 0
        Lv = 2.5*10^6.
        Rv = 461.

        # calculate the sum of qv at next
        sum_tracers_n = qv_Global_Integral_to_e(vert_coord, mesh, atmo_data, grid_tracers_n, grid_ps_n)
        q_sum = (sum_tracers_p - sum_tracers_n) # the unit is Pa !!!!!
        @info "previous e_p - e_n" q_sum
        # if (q_sum > 0)
        # es_sum = Mass_Weighted_Global_Integral(vert_coord, mesh, atmo_data, es, grid_ps_n) * 9.81 * 128*64*20
        #@info size(es) size(es_sum) size(q_sum)
        

        third_dim_size = 1
        weight = permutedims(repeat(mesh.wts ./ 2, inner=(1, mesh.nλ)), (2, 1)) ./ 128
        weight_new = repeat(reshape(weight, (size(weight, 1), size(weight, 2), 1)), 1, 1, third_dim_size) ./ third_dim_size

        e = zeros(128, 64, third_dim_size)
        es = zeros(128, 64, third_dim_size)

        for k in 1:third_dim_size
            e[:,:,k] = qv_to_e_2D(grid_tracers_n[:,:,k], grid_p_full[:,:,k])
            es[:,:,k] = 611.12 .* exp.(Lv ./ Rv .* (1. ./ 273.15 .- 1. ./ grid_t[:,:,k]))
        end
        e .+= (es .* weight_new ./ sum(es .* weight_new)) .* q_sum
        
        # e_sfc = qv_to_e_2D(grid_tracers_n[:,:,1], grid_ps_n[:,:,1])
        # e_sfc .+= (es .* weight ./ sum(es .* weight)) .* q_sum

        # FIXME WARNING: If there is no C-C equation, the water vapor pressure would be larger than pressure which induces negative value of qv
        # @info "minimum1: " minimum(grid_tracers_n)
        for k in 1:third_dim_size
            grid_tracers_n[:,:,k] .= e_to_qv_2D(e[:,:,k], grid_p_full[:,:,k])
        end
        # grid_tracers_n[:,:,1] .= e_to_qv_2D(e_sfc, grid_ps_n[:,:,1])
        # grid_tracers_n[grid_tracers_n .< 0] .= 0
        # @info "minimum2: " minimum(grid_tracers_n)
        # end
        # After correction, do the new sum of qv and tracers_ratio at next
        sum_tracers_n = qv_Global_Integral_to_e(vert_coord, mesh, atmo_data, grid_tracers_n, grid_ps_n)
        
        # @info "Sum after correction" sum_tracers_n
        tracers_ratio   = sum_tracers_p / sum_tracers_n
        @info "Correction Part (Not necessarily adding water): " ((sum_tracers_p - sum_tracers_n) / (4. .* pi .* atmo_data.radius^2)) (tracers_ratio)
        """
        grid_tracers_p[grid_tracers_p .< 0] .= 0
        grid_tracers_n[grid_tracers_n .< 0] .= 0
        Lv = 2.5*10^6.
        Rv = 461.

        # calculate the sum of qv at next
        sum_tracers_p_tmp =  sum_tracers_p ./ (4. * pi * atmo_data.radius^2)
        sum_tracers_n = qv_Global_Integral_to_e(vert_coord, mesh, atmo_data, grid_tracers_n, grid_ps_n) ./ (4. * pi * atmo_data.radius^2)
        q_sum = (sum_tracers_p_tmp - sum_tracers_n) # the unit is Pa !!!!!
        @info "previous e_p - e_n" q_sum
        # if (q_sum > 0)
        # es_sum = Mass_Weighted_Global_Integral(vert_coord, mesh, atmo_data, es, grid_ps_n) * 9.81 * 128*64*20
        #@info size(es) size(es_sum) size(q_sum)
        

        e = qv_to_e_2D(grid_tracers_n[:,:,1], grid_p_full[:,:,1])
        e .+=  q_sum
        grid_tracers_n[:,:,1] .= e_to_qv_2D(e, grid_p_full[:,:,1])

        sum_tracers_n = qv_Global_Integral_to_e(vert_coord, mesh, atmo_data, grid_tracers_n, grid_ps_n) ./ (4. * pi * atmo_data.radius^2)
        
        # @info "Sum after correction" sum_tracers_n
        tracers_ratio   = sum_tracers_p_tmp / sum_tracers_n
        @info "Correction Part (Not necessarily adding water): " (sum_tracers_p_tmp - sum_tracers_n) (tracers_ratio)

    end
    
end 





"""
compute vertical mass flux and velocity 
grid_M_half[:,:,k+1] = downward mass flux/per unit area across the K+1/2
grid_w_full[:,:,k]   = dp/dt vertical velocity 

update residuals
grid_δps[:,:,k]  += -∑_{r=1}^nd Dr = -∑_{r=1}^nd ∇(vrΔp_r)
grid_δt[:,:,k]  += κTw/p 
(grid_δu[:,:,k], grid_δv[:,:,k]) -= RT ∇p/p 

!  cell boundary. This is the "vertical velocity" in the hybrid coordinate system.
!  When vertical coordinate is pure sigma: grid_M_half = grid_ps*d(sigma)/dt
"""

function Four_In_One!(vert_coord::Vert_Coordinate, atmo_data::Atmo_Data, 
    grid_div::Array{Float64,3}, grid_u::Array{Float64,3}, grid_v::Array{Float64,3}, 
    grid_ps::Array{Float64,3},  grid_Δp::Array{Float64,3}, grid_lnp_half::Array{Float64,3}, grid_lnp_full::Array{Float64,3}, grid_p_full::Array{Float64,3},
    grid_dλ_ps::Array{Float64,3}, grid_dθ_ps::Array{Float64,3}, 
    grid_t::Array{Float64,3}, 
    grid_M_half::Array{Float64,3}, grid_w_full::Array{Float64,3}, 
    grid_δu::Array{Float64,3}, grid_δv::Array{Float64,3}, grid_δps::Array{Float64,3}, grid_δt::Array{Float64,3})
    
    rdgas, cp_air = atmo_data.rdgas, atmo_data.cp_air
    nd, bk = vert_coord.nd, vert_coord.bk
    Δak, Δbk = vert_coord.Δak, vert_coord.Δbk
    vert_difference_option = vert_coord.vert_difference_option
    
    kappa = rdgas / cp_air
    
    # dmean_tot = ∇ ∑_{k=1}^{nd} vk Δp_k = ∑_{k=1}^{nd} Dk
    nλ, nθ, _ = size(grid_ps)
    dmean_tot = zeros(Float64, nλ, nθ)
    Δlnp_p = zeros(Float64, nλ, nθ)
    Δlnp_m = zeros(Float64, nλ, nθ)
    Δlnp = zeros(Float64, nλ, nθ)
    x1 = zeros(Float64, nλ, nθ)
    dlnp_dλ = zeros(Float64, nλ, nθ)
    dlnp_dθ = zeros(Float64, nλ, nθ)
    dmean = zeros(Float64, nλ, nθ)
    x5 = zeros(Float64, nλ, nθ)
        
    if (vert_difference_option == "simmons_and_burridge") 
        for k = 1:nd
        Δp = grid_Δp[:,:,k]
        
        Δlnp_p .= grid_lnp_half[:,:,k + 1] - grid_lnp_full[:,:,k]
        Δlnp_m .= grid_lnp_full[:,:,k]   - grid_lnp_half[:,:,k]
        Δlnp   .= grid_lnp_half[:,:,k + 1] - grid_lnp_half[:,:,k]
        
        # angular momentum conservation 
        #    ∇p_k/p =  [(lnp_k - lnp_{k-1/2})∇p_{k-1/2} + (lnp_{k+1/2} - lnp_k)∇p_{k+1/2}]/Δpk
        #         =  [(lnp_k - lnp_{k-1/2})B_{k-1/2} + (lnp_{k+1/2} - lnp_k)B_{k+1/2}]/Δpk * ∇ps
        #         =  x1 * ∇ps
        x1 .= (bk[k] * Δlnp_m + bk[k + 1] * Δlnp_p ) ./ Δp
        
        dlnp_dλ .= x1 .* grid_dλ_ps[:,:,1]
        dlnp_dθ .= x1 .* grid_dθ_ps[:,:,1]
        
        
        
        # (grid_δu, grid_δv) -= RT ∇p/p 
        grid_δu[:,:,k] .-=  rdgas * grid_t[:,:,k] .* dlnp_dλ
        grid_δv[:,:,k] .-=  rdgas * grid_t[:,:,k] .* dlnp_dθ
        
        # dmean = ∇ (vk Δp_k) =  divk Δp_k + vk  Δbk[k] ∇ p_s
        dmean .= grid_div[:,:,k] .* Δp + Δbk[k] * (grid_u[:,:,k] .* grid_dλ_ps[:,:,1] + grid_v[:,:,k] .* grid_dθ_ps[:,:,1])
        
    
        # energy conservation for temperature
        # w/p = dlnp/dt = ∂lnp/∂t + dσ ∂lnp/∂σ + v∇lnp
        # dσ ∂ξ_k/∂σ = [M_{k+1/2}(ξ_k+1/2 - ξ_k) + M_{k-1/2}(ξ_k - ξ_k-1/2)]/Δp_k
        # weight the same way (TODO)
        # vertical advection operator (M is the downward speed)
        # dσ ∂lnp_k/∂σ = [M_{k+1/2}(lnp_k+1/2 - lnp_k) + M_{k-1/2}(lnp_k - lnp_k-1/2)]/Δp_k
        # ∂lnp/∂t = 1/p ∂p/∂t = [∂p/∂t_{k+1/2}(lnp_k+1/2 - lnp_k) + ∂p/∂t_{k-1/2}(lnp_k - lnp_k-1/2)]/Δp_k
        # As we know
        # ∂p/∂t_{k+1/2} = -∑_{r=1}^k Dr - M_{k+1/2}
        
        # ∂lnp/∂t + dσ ∂lnp/∂σ =  [(-∑_{r=1}^k Dr)(lnp_k+1/2 - lnp_k) + (-∑_{r=1}^{k-1} Dr)(lnp_k - lnp_k-1/2)]/Δp_k
        #                      = -[(∑_{r=1}^{k-1} Dr)(lnp_k+1/2 - lnp_k-1/2) + D_k(lnp_k+1/2 - lnp_k)]/Δp_k
        
        x5 .= -(dmean_tot .* Δlnp + dmean .* Δlnp_p) ./ Δp .+ grid_u[:,:,k] .* dlnp_dλ + grid_v[:,:,k] .* dlnp_dθ
        # grid_δt += κT w/p
        grid_δt[:,:,k] .+=  kappa * grid_t[:,:,k] .* x5
        # grid_w_full = w
        grid_w_full[:,:,k] .= x5 .* grid_p_full[:,:,k]
        # update dmean_tot to ∑_{r=1}^k ∇(vrΔp_r)
        dmean_tot .+= dmean
        # M_{k+1/2} = -∑_{r=1}^k ∇(vrΔp_r) - B_{k+1/2}∂ps/∂t
        grid_M_half[:,:,k + 1] .= -dmean_tot
        end
        
    else
        error("vert_difference_option ", vert_difference_option, " is not a valid value for option")
        
    end
    # ∂ps/∂t = -∑_{r=1}^nd ∇(vrΔp_r) = -dmean_tot
    grid_δps[:,:,1] .-= dmean_tot
    
    for k = 1:nd-1
        # M_{k+1/2} = -∑_{r=1}^k ∇(vrΔp_r) - B_{k+1/2}∂ps/∂t
        grid_M_half[:,:,k+1] .+= dmean_tot * bk[k+1]
    end
    
    grid_M_half[:,:,1] .= 0.0
    grid_M_half[:,:,nd + 1] .= 0.0
end 



"""
The governing equations are
∂div/∂t = ∇ × (A, B) - ∇^2E := f^d                    
∂lnps/∂t= (-∑_k div_k Δp_k + v_k ∇ Δp_k)/ps := f^p    
∂T/∂t = -(u,v)∇T - dσ∂T∂σ + κTw/p + J:= f^t           
Φ = f^Φ                                               

implicit part: -∇^2Φ - ∇(RT∇lnp) ≈ I^d = -∇^2(γT + H2 ps_ref lnps) - ∇^2 H1 ps_ref lnps, here RT∇lnp ≈  H1 ps_ref ∇lnps
implicit part:  f^p              ≈ I^p = -ν div / ps_ref
implicit part:  - dσ∂T∂σ + κTw/p ≈ I^t = -τ div  
implicit part:  f^Φ              ≈ I^Φ = γT + H2 ps_ref lnps 

We have 
δdiv = f^d - I^d + I^d
δlnps = f^p - I^p + I^p
δT = f^t - I^t + I^t

"""

function Spectral_Dynamics!(mesh::Spectral_Spherical_Mesh,  vert_coord::Vert_Coordinate, 
    atmo_data::Atmo_Data, dyn_data::Dyn_Data, 
    semi_implicit::Semi_Implicit_Solver)
    
    # spectral equation quantities
    spe_lnps_p, spe_lnps_c, spe_lnps_n, spe_δlnps = dyn_data.spe_lnps_p, dyn_data.spe_lnps_c, dyn_data.spe_lnps_n, dyn_data.spe_δlnps
    spe_vor_p, spe_vor_c, spe_vor_n, spe_δvor = dyn_data.spe_vor_p, dyn_data.spe_vor_c, dyn_data.spe_vor_n, dyn_data.spe_δvor
    spe_div_p, spe_div_c, spe_div_n, spe_δdiv = dyn_data.spe_div_p, dyn_data.spe_div_c, dyn_data.spe_div_n, dyn_data.spe_δdiv
    spe_t_p, spe_t_c, spe_t_n, spe_δt = dyn_data.spe_t_p, dyn_data.spe_t_c, dyn_data.spe_t_n, dyn_data.spe_δt
    
    # grid quantities
    grid_u_p, grid_u, grid_u_n = dyn_data.grid_u_p, dyn_data.grid_u_c, dyn_data.grid_u_n
    grid_v_p, grid_v, grid_v_n = dyn_data.grid_v_p, dyn_data.grid_v_c, dyn_data.grid_v_n
    grid_ps_p, grid_ps, grid_ps_n = dyn_data.grid_ps_p, dyn_data.grid_ps_c, dyn_data.grid_ps_n
    grid_t_p, grid_t, grid_t_n = dyn_data.grid_t_p, dyn_data.grid_t_c, dyn_data.grid_t_n


    # related quanties
    grid_p_half, grid_lnp_half, grid_p_full, grid_lnp_full = dyn_data.grid_p_half, dyn_data.grid_lnp_half, dyn_data.grid_p_full, dyn_data.grid_lnp_full
    grid_dλ_ps, grid_dθ_ps = dyn_data.grid_dλ_ps, dyn_data.grid_dθ_ps
    grid_lnps = dyn_data.grid_lnps
    
    grid_div, grid_absvor, grid_vor = dyn_data.grid_div, dyn_data.grid_absvor, dyn_data.grid_vor
    grid_w_full, grid_M_half = dyn_data.grid_w_full, dyn_data.grid_M_half
    grid_geopots, grid_geopot_full, grid_geopot_half = dyn_data.grid_geopots, dyn_data.grid_geopot_full, dyn_data.grid_geopot_half
    
    grid_energy_full, spe_energy = dyn_data.grid_energy_full, dyn_data.spe_energy
    

    grid_tracers_n = dyn_data.grid_tracers_n
    grid_tracers_c = dyn_data.grid_tracers_c
    grid_tracers_p = dyn_data.grid_tracers_p 
    # grid_δtracers  = dyn_data.grid_δtracers
    grid_δtracers  = zeros(Float64, mesh.nλ, mesh.nθ, 20)
    grid_tracers_full = dyn_data.grid_tracers_full

    ###
    # todo !!!!!!!!
    #  grid_q = grid_t
    
    # original 
    # pressure difference
    grid_Δp = dyn_data.grid_Δp
    # temporary variables
    grid_δQ = dyn_data.grid_d_full1
    
        
    # incremental quantities
    grid_δu, grid_δv, grid_δps, grid_δlnps, grid_δt = dyn_data.grid_δu, dyn_data.grid_δv, dyn_data.grid_δps, dyn_data.grid_δlnps, dyn_data.grid_δt

    integrator = semi_implicit.integrator
    Δt = Get_Δt(integrator)

    # Calculate latent heat and modify qv_current
    # HS_forcing_water_vapor!(grid_tracers_c,  grid_δtracers, grid_t, grid_δt, grid_p_full)
    """
    mean_ps_p, mean_energy_p, sum_tracers_p = Compute_Corrections_Init(
        vert_coord, mesh, atmo_data,
        grid_u_p, grid_v_p, grid_ps_p, grid_t_p, 
        grid_δu, grid_δv, grid_δt,  
        Δt, grid_energy_full, grid_tracers_p, grid_tracers_c, grid_δtracers, grid_tracers_full
    )
    
    # compute pressure based on grid_ps -> grid_p_half, grid_lnp_half, grid_p_full, grid_lnp_full 
    Pressure_Variables!(vert_coord, grid_ps, grid_p_half, grid_Δp, grid_lnp_half, grid_p_full, grid_lnp_full)

    
    
    # compute ∇ps = ∇lnps * ps
    Compute_Gradients!(mesh, spe_lnps_c,  grid_dλ_ps, grid_dθ_ps)
    grid_dλ_ps .*= grid_ps
    grid_dθ_ps .*= grid_ps


    
    # compute grid_M_half, grid_w_full, grid_δu, grid_δv, grid_δps, grid_δt, 
    # except the contributions from geopotential or vertical advection
    Four_In_One!(
        vert_coord, atmo_data, grid_div, grid_u, grid_v, grid_ps, 
        grid_Δp, grid_lnp_half, grid_lnp_full, grid_p_full,
        grid_dλ_ps, grid_dθ_ps, 
        grid_t, 
        grid_M_half, grid_w_full, grid_δu, grid_δv, grid_δps, grid_δt
    )

    Compute_Geopotential!(
        vert_coord, atmo_data, 
        grid_lnp_half, grid_lnp_full,  
        grid_t, 
        grid_geopots, grid_geopot_full, grid_geopot_half
    )
    

    grid_δlnps .= grid_δps ./ grid_ps
    Trans_Grid_To_Spherical!(mesh, grid_δlnps, spe_δlnps)
    
    
    # compute vertical advection, todo  finite volume method 
    Vert_Advection!(vert_coord, grid_u, grid_Δp, grid_M_half, Δt, vert_coord.vert_advect_scheme, grid_δQ)
    grid_δu  .+= grid_δQ
    Vert_Advection!(vert_coord, grid_v, grid_Δp, grid_M_half, Δt, vert_coord.vert_advect_scheme, grid_δQ)
    grid_δv  .+= grid_δQ
    Vert_Advection!(vert_coord, grid_t, grid_Δp, grid_M_half, Δt, vert_coord.vert_advect_scheme, grid_δQ)
    grid_δt  .+= grid_δQ
    # Tracer vertical advection
    # Vert_Advection!(vert_coord, grid_tracers_c, grid_Δp, grid_M_half, Δt, vert_coord.vert_advect_scheme,  grid_δQ)
    # grid_δtracers .+= grid_δQ


    Add_Horizontal_Advection!(mesh, spe_t_c, grid_u, grid_v, grid_δt)
    Trans_Grid_To_Spherical!(mesh, grid_δt, spe_δt)
    
   

    grid_absvor = dyn_data.grid_absvor
    Compute_Abs_Vor!(grid_vor, atmo_data.coriolis, grid_absvor)
    
    
    grid_δu .+=  grid_absvor .* grid_v
    grid_δv .-=  grid_absvor .* grid_u
    
    
    Vor_Div_From_Grid_UV!(mesh, grid_δu, grid_δv, spe_δvor, spe_δdiv)
    

    grid_energy_full .= grid_geopot_full .+ 0.5 * (grid_u.^2 + grid_v.^2)
    Trans_Grid_To_Spherical!(mesh, grid_energy_full, spe_energy)
    Apply_Laplacian!(mesh, spe_energy)
    spe_δdiv .-= spe_energy
    
    
    Implicit_Correction!(semi_implicit, vert_coord, atmo_data,
    spe_div_c, spe_div_p, spe_lnps_c, spe_lnps_p, spe_t_c, spe_t_p, 
    spe_δdiv, spe_δlnps, spe_δt)
    
    Compute_Spectral_Damping!(integrator, spe_vor_c, spe_vor_p, spe_δvor)
    Compute_Spectral_Damping!(integrator, spe_div_c, spe_div_p, spe_δdiv)
    Compute_Spectral_Damping!(integrator, spe_t_c, spe_t_p, spe_δt)

        
    Filtered_Leapfrog!(integrator, spe_δvor, spe_vor_p, spe_vor_c, spe_vor_n)
    Filtered_Leapfrog!(integrator, spe_δdiv, spe_div_p, spe_div_c, spe_div_n)
    Filtered_Leapfrog!(integrator, spe_δlnps, spe_lnps_p, spe_lnps_c, spe_lnps_n)
    
    
    Trans_Spherical_To_Grid!(mesh, spe_vor_n, grid_vor)
    Trans_Spherical_To_Grid!(mesh, spe_div_n, grid_div)
    UV_Grid_From_Vor_Div!(mesh, spe_vor_n, spe_div_n, grid_u_n, grid_v_n)
    Trans_Spherical_To_Grid!(mesh, spe_lnps_n, grid_lnps)
    grid_ps_n .= exp.(grid_lnps)

    """
    ### By CJY 0517
    # HS_forcing_water_vapor!(grid_tracers_c,  grid_δtracers, grid_t, grid_δt, grid_p_full)
    ###

     #TODO: Tracer horizontal advection
     Tracer_Horizontal_Advection!(mesh, grid_tracers_c, grid_δQ, grid_u, grid_v)
     grid_δtracers .+= grid_δQ

    # Tracers leapfrog
    init_step = integrator.init_step
    robert_coef = integrator.robert_coef
    Δt = integrator.Δt
    if (init_step) 
        grid_tracers_n .= grid_tracers_c + Δt*grid_δtracers
        grid_tracers_c .+= robert_coef*(-1.0*grid_tracers_c + grid_tracers_n)
    else
        grid_tracers_c .+= robert_coef*(grid_tracers_p - 2*grid_tracers_c)
        grid_tracers_n .= grid_tracers_p + 2*Δt*grid_δtracers
        grid_tracers_c .+= robert_coef*grid_tracers_n
    end

    ##### Diffusion
    # dx = (mesh.λc[3] - mesh.λc[1]) * mesh.radius / 2.
    # dy = (mesh.θc[3] - mesh.θc[1]) * mesh.radius / 2.
    # KX = 1000.
    # KY = 1000.

    # grid_tracers_diffusion_x = zeros(Float64, mesh.nλ, mesh.nθ, mesh.nd)
    # grid_tracers_diffusion_y = zeros(Float64, mesh.nλ, mesh.nθ, mesh.nd)
    # for k in 1:20
    #     for j in 1:64
    #         grid_tracers_diffusion_x[:,j,k] = 2*Δt*KX * (circshift(grid_tracers_p[:, j, k], -1) .- 2. * grid_tracers_p[:, j, k] .+ circshift(grid_tracers_p[:, j, k], 1)) ./ (dx * cos(abs(mesh.θc[j])))^2
    #     end
    # end

    # for k in 1:20
    #     for i in 1:128
    #         grid_tracers_diffusion_y[i,:,k] = 2*Δt*KY * (circshift(grid_tracers_p[i, :, k], -1) .- 2. * grid_tracers_p[i, :, k] .+ circshift(grid_tracers_p[i, :, k], 1)) ./ dy^2
    #     end
    # end

    # grid_tracers_n .+= grid_tracers_diffusion_x .+ grid_tracers_diffusion_y

    # Filtered_Leapfrog!(integrator, spe_δt, spe_t_p, spe_t_c, spe_t_n)
    # Trans_Spherical_To_Grid!(mesh, spe_t_n, grid_t_n)


    # Compute_Corrections!(
    #     vert_coord, mesh, atmo_data, mean_ps_p, mean_energy_p, 
    #     grid_u_n, grid_v_n,
    #     grid_energy_full,
    #     grid_ps_n, spe_lnps_n, 
    #     grid_t_n, spe_t_n, 
    #     sum_tracers_p, grid_tracers_p, grid_tracers_c, grid_tracers_n,
    #     grid_t,
    #     grid_p_full
    # )

    ###

    day_to_sec = 86400
    if (integrator.time%day_to_sec == 0)
        @info "Day: ", div(integrator.time,day_to_sec), " Max |U|,|V|,|P|,|T|,|qv|: ", maximum(abs.(dyn_data.grid_u_c)), maximum(abs.(dyn_data.grid_v_c)), maximum(dyn_data.grid_p_full), maximum(dyn_data.grid_t_c), maximum(dyn_data.grid_tracers_c)
        @info "Day: ", div(integrator.time,day_to_sec), " Min |U|,|V|,|P|,|T|,|qv|: ", minimum(abs.(dyn_data.grid_u_c)), minimum(abs.(dyn_data.grid_v_c)), minimum(dyn_data.grid_p_full), minimum(dyn_data.grid_t_c), minimum(dyn_data.grid_tracers_c)
    end
    # @info "Time: ", integrator.time, " Max |U|,|V|,|P|,|T|,|qv|: ", maximum(abs.(dyn_data.grid_u_c)), maximum(abs.(dyn_data.grid_v_c)), maximum(dyn_data.grid_p_full), maximum(dyn_data.grid_t_c), maximum(dyn_data.grid_tracers_c)
    # @info "Time: ", integrator.time, " Min |U|,|V|,|P|,|T|,|qv|: ", minimum(abs.(dyn_data.grid_u_c)), minimum(abs.(dyn_data.grid_v_c)), minimum(dyn_data.grid_p_full), minimum(dyn_data.grid_t_c), minimum(dyn_data.grid_tracers_c)


    Time_Advance!(dyn_data)

    # Pressure_Variables!(vert_coord, grid_ps, grid_p_half, grid_Δp, grid_lnp_half, grid_p_full, grid_lnp_full)
    
    return 
end 

function Get_Topography!(grid_geopots::Array{Float64, 3})
    #####grid_geopots .= 0.0
    read_file = load("0404_300_50_8_variables.dat")
    grid_geopots .= read_file["grid_geopots_xyzt"][:,:,1,300]
    return
end 


function Tracer_Horizontal_Advection!(mesh::Spectral_Spherical_Mesh, grid_tracer, grid_δQ, grid_u, grid_v)
    # WARNING d2x should multiply cos(θ)
    d2x = abs(mesh.λc[3] - mesh.λc[1]) * mesh.radius 
    d2y = abs(mesh.θc[3] - mesh.θc[1]) * mesh.radius 
    dy_bd = abs(mesh.θc[63] - mesh.θc[64]) * mesh.radius 

    pgrid_px = zeros(Float64, mesh.nλ, mesh.nθ, 20)
    pgrid_py = zeros(Float64, mesh.nλ, mesh.nθ, 20)
    for k in 1:20
        for j in 1:64
            pgrid_px[:, j, k] .= ((circshift(grid_tracer[:, j, k], -1).*circshift(grid_u[:, j, k], -1)) .- (circshift(grid_tracer[:, j, k], 1).*circshift(grid_u[:, j, k], 1))) ./ (d2x * cos(abs(mesh.θc[j])))
        end
    end
    # @info "pgrid_px" maximum(pgrid_px) minimum(pgrid_px)

    # Zonal advection
    for k in 1:2
        for i in 1:mesh.nλ
            pgrid_py[i, 2:63, k] .= (circshift(grid_tracer[i, :, k], -1)[2:63] .* circshift(grid_v[i, :, k], -1)[2:63] .- circshift(grid_tracer[i, :, k], 1)[2:63] .* circshift(grid_v[i, :, k], 1)[2:63]) ./ (d2y)
        end

        # Option 1:
        # Zonal advection and solving boundary using average of \frac{\partial (vq)}{\partial y} and allocate it to every points.

        # pgrid_py[1:64, 64, k] .= ((grid_tracer[65:128, 64, k] .* -grid_v[65:128, 64, k]) .- (grid_tracer[1:64, 63, k] .* grid_v[1:64, 63, k])) ./ (dy_bd * 2.)
        # pgrid_py[65:128, 64, k] .= ((grid_tracer[1:64, 64, k] .* -grid_v[1:64, 64, k]) .- (grid_tracer[65:128, 63, k] .* grid_v[65:128, 63, k])) ./ (dy_bd * 2.)
        # pgrid_py[:,64,k] .= mean(pgrid_py[:,64,k] / (dy_bd * 2.))
        # pgrid_py[:,1,k] .= mean(pgrid_py[:,1,k] / (dy_bd * 2.))

        
        # Option 2:
        # Zonal advection and Solving boundary using ∇^2 q_{inf} = f(x), f(x) = \frac{\partial (vq)}{\partial y}
        pgrid_py[1:64, 64, k] .= ((grid_tracer[65:128, 64, k] .* -grid_v[65:128, 64, k]) .- (grid_tracer[1:64, 63, k] .* grid_v[1:64, 63, k])) 
        pgrid_py[65:128, 64, k] .= ((grid_tracer[1:64, 64, k] .* -grid_v[1:64, 64, k]) .- (grid_tracer[65:128, 63, k] .* grid_v[65:128, 63, k])) 

        A = zeros(Float64, 128+2, 128+2)
        for i in 1:130
            A[i, i] = 2.
            if i != 1 && i != 130
                A[i, i - 1] = -1.
                A[i, i + 1] = -1.
            elseif i == 1
                A[i, i + 1] = -1.
            else 
                A[i, i - 1] = -1.
            end
        end
        A = sparse(A)

        b = zeros(Float64, 128+2)
        b[1] = 0.
        b[130] = 0.
        b[2:129] .= grid_tracer[:,64,k]
        b .= (d2x / 2. * cos(abs(mesh.θc[64]))) ./ 2. .* b

        x1 = IterativeSolvers.bicgstabl(A, b, max_mv_products=5000)
        pgrid_py[:,64,k] .= x1[2:129]

        b2 = zeros(Float64, 128+2)
        b2[1] = 0.
        b2[130] = 0.
        b2[2:129] .= grid_tracer[:,1,k]
        b2 .= (d2x / 2. * cos(abs(mesh.θc[64]))) ./ 2. .* b2

        x2 = IterativeSolvers.bicgstabl(A, b2, max_mv_products=5000)
        pgrid_py[:,1,k] .= x2[2:129]
    end
    # @info maximum(pgrid_py[:, 64, 1])



    grid_δQ .= -(pgrid_px + pgrid_py)
    # grid_δQ .= -(pgrid_py)
end

function Spectral_Initialize_Fields!(mesh::Spectral_Spherical_Mesh, atmo_data::Atmo_Data, vert_coord::Vert_Coordinate, sea_level_ps_ref::Float64, init_t::Float64,
    grid_geopots::Array{Float64,3}, dyn_data::Dyn_Data)
    
    spe_vor_c, spe_div_c, spe_lnps_c, spe_t_c = dyn_data.spe_vor_c, dyn_data.spe_div_c, dyn_data.spe_lnps_c, dyn_data.spe_t_c
    spe_vor_p, spe_div_p, spe_lnps_p, spe_t_p = dyn_data.spe_vor_p, dyn_data.spe_div_p, dyn_data.spe_lnps_p, dyn_data.spe_t_p
    grid_u, grid_v, grid_ps, grid_t = dyn_data.grid_u_c, dyn_data.grid_v_c, dyn_data.grid_ps_c, dyn_data.grid_t_c
    grid_u_p, grid_v_p, grid_ps_p, grid_t_p = dyn_data.grid_u_p, dyn_data.grid_v_p, dyn_data.grid_ps_p, dyn_data.grid_t_p
    
    grid_lnps,  grid_vor, grid_div =  dyn_data.grid_lnps, dyn_data.grid_vor, dyn_data.grid_div
    
    grid_p_half, grid_Δp, grid_lnp_half, grid_p_full, grid_lnp_full = dyn_data.grid_p_half, dyn_data.grid_Δp, dyn_data.grid_lnp_half, dyn_data.grid_p_full, dyn_data.grid_lnp_full
    nλ, nθ, nd = mesh.nλ, mesh.nθ, mesh.nd
    
    ### By CJY2
        
    grid_tracers_n = dyn_data.grid_tracers_n
    grid_tracers_c = dyn_data.grid_tracers_c
    grid_tracers_p = dyn_data.grid_tracers_p 
    ###

    
    rdgas = atmo_data.rdgas
    #grid_t    .= init_t
    read_file = load("0404_300_50_8_variables.dat")
    initial_day = 300
    grid_t    .= read_file["grid_t_c_xyzt"][:,:,:,initial_day] 
    # dΦ/dlnp = -RT    Δp = -ΔΦ/RT
    grid_geopots .= read_file["grid_geopots_xyzt"][:,:,:,initial_day]
    #grid_lnps .= log(sea_level_ps_ref) .- grid_geopots / (rdgas * init_t)
    grid_lnps .= read_file["grid_lnps_xyzt"][:,:,1,initial_day]
    #grid_ps   .= exp.(grid_lnps)
    grid_ps    .= read_file["grid_ps_xyzt"][:,:,1,initial_day]
    

    # By CJY
    #spe_div_c .= 0.0
    #spe_vor_c .= 0.0

    # # initial perturbation
    num_fourier, num_spherical = mesh.num_fourier, mesh.num_spherical
    """
    initial_perturbation = 1.0e-7/sqrt(2.0)
    # initial vorticity perturbation used in benchmark code
    # In gfdl spe[i,j] =  myspe[i, i+j-1]*√2

    for k = nd-2:nd
        spe_vor_c[2,5,k] = initial_perturbation
        spe_vor_c[6,9,k] = initial_perturbation
        spe_vor_c[2,4,k] = initial_perturbation  
        spe_vor_c[6,8,k] = initial_perturbation
    end
    """

    ###
    spe_vor_c[:,:,:] .= read_file["spe_vor_c_xyzt"][:,:,:,initial_day]
    spe_div_c[:,:,:] .= read_file["spe_div_c_xyzt"][:,:,:,initial_day]
    # grid_u[:,:,:]    .= read_file["grid_u_c_xyzt"][:,:,:,initial_day]
    # grid_v[:,:,:]    .= read_file["grid_v_c_xyzt"][:,:,:,initial_day]  
    
    
    UV_Grid_From_Vor_Div!(mesh, spe_vor_c, spe_div_c, grid_u, grid_v)
    
    # initial spectral fields (and spectrally-filtered) grid fields
    Trans_Grid_To_Spherical!(mesh, grid_t, spe_t_c)
    Trans_Spherical_To_Grid!(mesh, spe_t_c, grid_t)

    Trans_Grid_To_Spherical!(mesh, grid_lnps, spe_lnps_c)
    Trans_Spherical_To_Grid!(mesh, spe_lnps_c,  grid_lnps)

    # By CJY ### grid_ps .= exp.(grid_lnps)
    #grid_ps .= read_file["grid_ps_xyzt"][:,:,1,90]
    grid_ps .= exp.(grid_lnps)
    
    Vor_Div_From_Grid_UV!(mesh, grid_u, grid_v, spe_vor_c, spe_div_c)

    UV_Grid_From_Vor_Div!(mesh, spe_vor_c, spe_div_c, grid_u, grid_v)
    
    Trans_Spherical_To_Grid!(mesh, spe_vor_c, grid_vor)
    Trans_Spherical_To_Grid!(mesh, spe_div_c, grid_div)

    #update pressure variables for hs forcing
    grid_p_half .= read_file["grid_p_half_xyzt"][:,:,:,initial_day]
    grid_Δp .= read_file["grid_Δp_xyzt"][:,:,:,initial_day]
    grid_lnp_half .= read_file["grid_lnp_half_xyzt"][:,:,:,initial_day]
    grid_p_full .= read_file["grid_p_full_xyzt"][:,:,:,initial_day]
    grid_lnp_full .= read_file["grid_lnp_full_xyzt"][:,:,:,initial_day]
        
    Pressure_Variables!(vert_coord, grid_ps, grid_p_half, grid_Δp,
    grid_lnp_half, grid_p_full, grid_lnp_full)
    
    
    spe_vor_p .= spe_vor_c
    spe_div_p .= spe_div_c
    spe_lnps_p .= spe_lnps_c
    spe_t_p .= spe_t_c


    grid_u_p .= grid_u
    grid_v_p .= grid_v
    grid_ps_p .= grid_ps
    grid_t_p .= grid_t

    # Tracer initialization
    initial_RH = 0.5 
    Lv = 2.5*10^6.
    Rv = 461.
    # grid_tracers_c .= (0.622 .* (611.12 .* exp.(Lv ./ Rv .* (1. ./ 273.15 .- 1. ./ grid_t)) .* initial_RH)) ./ (grid_p_full .- 0.378 .* (611.12 .* exp.(Lv ./ Rv .* (1. ./ 273.15 .- 1. ./ grid_t)) .* initial_RH)) 


    # Advection test
    # tracer
    for k in 1:20
        for j in 1:64
            for i in 1:128
                grid_tracers_c[i,j,k] = Advection_tracer(mesh, mesh.λc[i], mesh.θc[j])
                grid_u[i,j,k] = AdvectionU(mesh, mesh.λc[i], mesh.θc[j])
                grid_v[i,j,k] = AdvectionV(mesh, mesh.λc[i])
                
            end
        end
    end

    grid_u_n, grid_v_n = dyn_data.grid_u_n, dyn_data.grid_v_n
    grid_u_p    .= grid_u
    grid_u_n    .= grid_u
    grid_v_p    .= grid_v
    grid_v_n    .= grid_v

    # By CJY2
    grid_tracers_p .= grid_tracers_c

end 

# Advection test
function Advection_tracer(mesh, lon, lat)
    h0 = 1000.
    lonC = 3. * pi / 2.
    latC = 0.
    rd = mesh.radius * acos(sin(latC) * sin(lat) + cos(latC) * cos(lat) * cos(lon-lonC))
    r0 = mesh.radius ./ 3.
    if (rd < r0) 
        return h0 / 2. * (1 + cos(pi * rd / r0))
    else 
        return 0.
    end
end

function AdvectionU(mesh, lon, lat) 
    ALPHA0 = pi / 2.
    u0 = 2. * pi * mesh.radius / (12. * 86400)
    u = u0 * (cos(ALPHA0) * cos(lat) + sin(ALPHA0) * cos(lon) * sin(lat))
    return u
end

function AdvectionV(mesh, lon) 
    ALPHA0 = pi / 2.
    u0 = 2. * pi * mesh.radius / (12. * 86400)
    v = - u0 * sin(ALPHA0) * sin(lon)
    return v
end


function Spectral_Dynamics_Physics!(atmo_data::Atmo_Data, mesh::Spectral_Spherical_Mesh, dyn_data::Dyn_Data, Δt::Int64, physics_params::Dict{String, Float64})
    grid_δu, grid_δv, grid_δps, grid_δt = dyn_data.grid_δu, dyn_data.grid_δv, dyn_data.grid_δps, dyn_data.grid_δt
    grid_u_p, grid_v_p,  grid_t_p = dyn_data.grid_u_p, dyn_data.grid_v_p, dyn_data.grid_t_p
    grid_p_half, grid_p_full = dyn_data.grid_p_half, dyn_data.grid_p_full
    grid_t_eq = dyn_data.grid_t_eq
    
    grid_δps .= 0.0

    HS_Forcing!(atmo_data, Δt, mesh.sinθ, grid_u_p, grid_v_p, grid_p_half, grid_p_full, grid_t_p, grid_δu, grid_δv,
    grid_t_eq, grid_δt, physics_params)

end


function Atmosphere_Update!(mesh::Spectral_Spherical_Mesh, atmo_data::Atmo_Data, vert_coord::Vert_Coordinate, semi_implicit::Semi_Implicit_Solver, 
                            dyn_data::Dyn_Data, physcis_params::Dict{String, Float64})

    # Δt = Get_Δt(semi_implicit.integrator)
    # Spectral_Dynamics_Physics!(atmo_data, mesh,  dyn_data, Δt, physcis_params)
    Spectral_Dynamics!(mesh,  vert_coord , atmo_data, dyn_data, semi_implicit)

    # grid_ps , grid_Δp, grid_p_half, grid_lnp_half, grid_p_full, grid_lnp_full = dyn_data.grid_ps_c,  dyn_data.grid_Δp, dyn_data.grid_p_half, 
    #                                                                             dyn_data.grid_lnp_half, dyn_data.grid_p_full, dyn_data.grid_lnp_full 
    # grid_t = dyn_data.grid_t_c
    # grid_geopots, grid_z_full, grid_z_half = dyn_data.grid_geopots, dyn_data.grid_z_full, dyn_data.grid_z_half
        
    # Compute_Pressures_And_Heights!(atmo_data, vert_coord,     
    # grid_ps, grid_geopots, grid_t, 
    # grid_p_half, grid_Δp, grid_lnp_half, grid_p_full, grid_lnp_full, grid_z_full, grid_z_half)

    return
end 


function HS_forcing_water_vapor!(grid_tracers_c::Array{Float64, 3}, grid_δtracers::Array{Float64, 3}, grid_t::Array{Float64, 3}, grid_δt::Array{Float64, 3}, grid_p_full::Array{Float64, 3})
    extra_e = zeros(size(grid_tracers_c)...)
    grid_e_from_qv = zeros(size(grid_tracers_c)...)

    cp  = 1004.
    Lv = 2.5*10^6.
    Rv = 461.
    
    es = 611.12 .* exp.(Lv ./ Rv .* (1. ./ 273.15 .- 1. ./ grid_t))


    for k in 1:20
        grid_e_from_qv[:,:,k] .= qv_to_e_2D(grid_tracers_c[:,:,k], grid_p_full[:,:,k])
    end

    # @info maximum(grid_e_from_qv)
    # @info minimum(grid_e_from_qv)

    for k = 1:20
        for j=1:64
            for i=1:128
                if grid_e_from_qv[i,j,k] >= es[i,j,k]
                    # @info "yes"
                    extra_e[i,j,k] = grid_e_from_qv[i,j,k] - es[i,j,k]
                    grid_e_from_qv[i,j,k] = es[i,j,k]
                else
                    grid_e_from_qv[i,j,k] = grid_tracers_c[i,j,k]
                end
            end
        end
    end

    for k in 1:20
        grid_tracers_c[:,:,k] .= e_to_qv_2D(grid_e_from_qv[:,:,k], grid_p_full[:,:,k])
    end

    # FIXME
    day_to_sec = 86400.
    grid_δt .+= (grid_tracers_c .* Lv ./ cp) ./ day_to_sec

end


function e_to_qv_2D(e::Array{Float64, 2}, grid_P::Array{Float64, 2})
    # if minimum(grid_P - 0.378 * e) .< 0
    #     @info "warning: e >> P"
    #     return (e .* 0.622) ./ (grid_P)
    # end
    return (e .* 0.622) ./ (grid_P - 0.378 * e)
end

function qv_to_e_2D(qv::Array{Float64, 2}, grid_P::Array{Float64, 2})
    return (qv .* grid_P) ./ (0.378 .* qv .+ 0.622)
end


# function Spectral_Dynamics_Main()
#   # the decay of a sinusoidal disturbance to a zonally symmetric flow 
#   # that resembles that found in the upper troposphere in Northern winter.
#   name = "Spectral_Dynamics"
#   #num_fourier, nθ, nd = 63, 96, 20
#   num_fourier, nθ, nd = 42, 64, 20
#   #num_fourier, nθ, nd = 21, 32, 20
#   num_spherical = num_fourier + 1
#   nλ = 2nθ
  
#   radius = 6371000.0
#   omega = 7.292e-5
#   sea_level_ps_ref = 1.0e5
#   init_t = 264.0
  
#   # Initialize mesh
#   mesh = Spectral_Spherical_Mesh(num_fourier, num_spherical, nθ, nλ, nd, radius)
#   θc, λc = mesh.θc,  mesh.λc
#   cosθ, sinθ = mesh.cosθ, mesh.sinθ
  
#   vert_coord = Vert_Coordinate(nλ, nθ, nd, "even_sigma", "simmons_and_burridge", "second_centered_wts", sea_level_ps_ref)
#   # Initialize atmo_data
#   do_mass_correction = true
#   do_energy_correction = true
#   do_water_correction = false
  
#   use_virtual_temperature = false
#   atmo_data = Atmo_Data(name, nλ, nθ, nd, do_mass_correction, do_energy_correction, do_water_correction, use_virtual_temperature, sinθ, radius,  omega)
  
#   # Initialize integrator
#   damping_order = 4
#   damping_coef = 1.15741e-4
#   robert_coef  = 0.04 
  
#   implicit_coef = 0.5
#   day_to_sec = 86400
#   start_time = 0
#   end_time = 2*day_to_sec  #
#   Δt = 1200
#   init_step = true
  
#   integrator = Filtered_Leapfrog(robert_coef, 
#   damping_order, damping_coef, mesh.laplacian_eig,
#   implicit_coef, Δt, init_step, start_time, end_time)
  
#   ps_ref = sea_level_ps_ref
#   t_ref = fill(300.0, nd)
#   wave_numbers = mesh.wave_numbers
#   semi_implicit = Semi_Implicit_Solver(vert_coord, atmo_data,
#   integrator, ps_ref, t_ref, wave_numbers)
  
#   # Initialize data
#   dyn_data = Dyn_Data(name, num_fourier, num_spherical, nλ, nθ, nd)
  
  
#   NT = Int64(end_time / Δt)
  
#   Get_Topography!(dyn_data.grid_geopots)
  
#   Spectral_Initialize_Fields!(mesh, atmo_data, vert_coord, sea_level_ps_ref, init_t,
#   dyn_data.grid_geopots, dyn_data)
  

#   Atmosphere_Update!(mesh, atmo_data, vert_coord, semi_implicit, dyn_data)

#   Update_Init_Step!(semi_implicit)
#   integrator.time += Δt 
#   for i = 2:NT

#     Atmosphere_Update!(mesh, atmo_data, vert_coord, semi_implicit, dyn_data)

#     integrator.time += Δt
#     @info integrator.time

#   end
  
# end


# #Spectral_Dynamics_Main()

