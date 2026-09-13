use proc_macro::TokenStream;
use quote::quote;
use syn::{
    FnArg, ItemFn, Lit, Meta, Token, parse::Parse, parse::ParseStream, parse_macro_input,
    punctuated::Punctuated,
};

struct TaskArgs {
    priority: String,
    affinity: String,
    kind: String,
    stack: String,
    switcher: String,
}

impl Parse for TaskArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let vars = Punctuated::<Meta, Token![,]>::parse_terminated(input)?;
        let mut priority = "Normal".to_string();
        let mut affinity = "SameCore".to_string();
        let mut kind = "Compute".to_string();
        let mut stack = "2M".to_string();
        let mut switcher = "CrossThreadFloat".to_string();

        for var in vars {
            if let Meta::NameValue(nv) = var {
                if nv.path.is_ident("priority") {
                    if let syn::Expr::Lit(syn::ExprLit {
                        lit: Lit::Str(s), ..
                    }) = nv.value
                    {
                        priority = s.value();
                    }
                } else if nv.path.is_ident("affinity") {
                    if let syn::Expr::Lit(syn::ExprLit {
                        lit: Lit::Str(s), ..
                    }) = nv.value
                    {
                        affinity = s.value();
                    }
                } else if nv.path.is_ident("kind") {
                    if let syn::Expr::Lit(syn::ExprLit {
                        lit: Lit::Str(s), ..
                    }) = nv.value
                    {
                        kind = s.value();
                    }
                } else if nv.path.is_ident("stack") {
                    if let syn::Expr::Lit(syn::ExprLit {
                        lit: Lit::Str(s), ..
                    }) = nv.value
                    {
                        stack = s.value();
                    }
                } else if nv.path.is_ident("switcher")
                    && let syn::Expr::Lit(syn::ExprLit {
                        lit: Lit::Str(s), ..
                    }) = nv.value
                {
                    switcher = s.value();
                }
            }
        }

        Ok(TaskArgs {
            priority,
            affinity,
            kind,
            stack,
            switcher,
        })
    }
}

#[proc_macro_attribute]
pub fn task(args: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(args as TaskArgs);
    let mut input = parse_macro_input!(item as ItemFn);

    let fn_name = &input.sig.ident;
    let priority = &args.priority;
    let affinity = &args.affinity;
    let kind = &args.kind;
    let stack = &args.stack;

    let metadata_mod = syn::Ident::new(&format!("dtact_metadata_{}", fn_name), fn_name.span());
    let priority_ident = syn::Ident::new(priority, fn_name.span());
    let affinity_ident = syn::Ident::new(affinity, fn_name.span());
    let kind_ident = syn::Ident::new(kind, fn_name.span());
    let switcher = &args.switcher;
    let switcher_ident = syn::Ident::new(switcher, fn_name.span());

    let return_type = match &input.sig.output {
        syn::ReturnType::Default => quote! { () },
        syn::ReturnType::Type(_, ty) => quote! { #ty },
    };

    input.sig.asyncness = None;
    input.sig.output = syn::parse2(quote! {
        -> dtact::api::TaskFuture<impl std::future::Future<Output = #return_type> + Send + 'static, dtact::#switcher_ident>
    }).unwrap();

    let vis = &input.vis;
    let attrs = &input.attrs;
    let sig = &input.sig;
    let body = &input.block;

    let expanded = quote! {
        #(#attrs)*
        #vis #sig {
            let fut = async move #body;
            dtact::api::TaskFuture {
                future: fut,
                priority: dtact::Priority::#priority_ident,
                affinity: dtact::topology::Affinity::#affinity_ident,
                kind: dtact::WorkloadKind::#kind_ident,
                _marker: std::marker::PhantomData,
            }
        }

        pub mod #metadata_mod {
            pub const PRIORITY: dtact::Priority = dtact::Priority::#priority_ident;
            pub const AFFINITY: dtact::topology::Affinity = dtact::topology::Affinity::#affinity_ident;
            pub const KIND: dtact::WorkloadKind = dtact::WorkloadKind::#kind_ident;
            pub const STACK_SIZE: &'static str = #stack;
            pub type SWITCHER = dtact::#switcher_ident;
        }
    };

    TokenStream::from(expanded)
}

#[proc_macro_attribute]
pub fn export_async(_args: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);
    let fn_name = &input.sig.ident;
    let wrapper_name = syn::Ident::new(&format!("dtact_export_{}", fn_name), fn_name.span());

    let mut c_params = Vec::new();
    let mut call_args = Vec::new();

    for input in &input.sig.inputs {
        if let FnArg::Typed(pat_type) = input {
            let pat = &pat_type.pat;
            let ty = &pat_type.ty;
            c_params.push(quote! { #pat: #ty });
            call_args.push(quote! { #pat });
        } else {
            panic!("export_async does not support 'self' parameters");
        }
    }

    let expanded = quote! {
        #input

        #[unsafe(no_mangle)]
        pub extern "C" fn #wrapper_name(#(#c_params),*) -> dtact::dtact_handle_t {
            dtact::spawn(#fn_name(#(#call_args),*))
        }
    };

    TokenStream::from(expanded)
}

#[proc_macro_attribute]
pub fn export_fiber(_args: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);
    let fn_name = &input.sig.ident;
    let wrapper_name = syn::Ident::new(&format!("dtact_export_fiber_{}", fn_name), fn_name.span());

    let mut c_params = Vec::new();
    let mut call_args = Vec::new();

    for input in &input.sig.inputs {
        if let FnArg::Typed(pat_type) = input {
            let pat = &pat_type.pat;
            let ty = &pat_type.ty;
            c_params.push(quote! { #pat: #ty });
            call_args.push(quote! { #pat });
        } else {
            panic!("export_fiber does not support 'self' parameters");
        }
    }

    let expanded = quote! {
        #input

        #[unsafe(no_mangle)]
        pub extern "C" fn #wrapper_name(#(#c_params),*) -> dtact::dtact_handle_t {
            dtact::api::fiber::spawn_with_stack("2M", move || {
                #fn_name(#(#call_args),*);
            })
        }
    };

    TokenStream::from(expanded)
}

struct InitArgs {
    topology: String,
    safety: String,
    workers: usize,
    capacity: u32,
    stack: usize,
    numa: usize,
}

impl Parse for InitArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let vars = Punctuated::<Meta, Token![,]>::parse_terminated(input)?;
        let mut topology = "P2PMesh".to_string();
        let mut safety = "Safety1".to_string();
        let mut workers = 0;
        let mut capacity = 4096;
        let mut stack = 512 * 1024;
        let mut numa = 0;

        for var in vars {
            if let Meta::NameValue(nv) = var {
                if nv.path.is_ident("topology") {
                    if let syn::Expr::Lit(syn::ExprLit {
                        lit: Lit::Str(s), ..
                    }) = &nv.value
                    {
                        topology = s.value();
                    }
                } else if nv.path.is_ident("safety") {
                    if let syn::Expr::Lit(syn::ExprLit {
                        lit: Lit::Str(s), ..
                    }) = &nv.value
                    {
                        safety = s.value();
                    }
                } else if nv.path.is_ident("workers") {
                    if let syn::Expr::Lit(syn::ExprLit {
                        lit: Lit::Int(i), ..
                    }) = &nv.value
                    {
                        workers = i.base10_parse()?;
                    }
                } else if nv.path.is_ident("capacity") {
                    if let syn::Expr::Lit(syn::ExprLit {
                        lit: Lit::Int(i), ..
                    }) = &nv.value
                    {
                        capacity = i.base10_parse()?;
                    }
                } else if nv.path.is_ident("stack") {
                    if let syn::Expr::Lit(syn::ExprLit {
                        lit: Lit::Int(i), ..
                    }) = &nv.value
                    {
                        stack = i.base10_parse()?;
                    }
                } else if nv.path.is_ident("numa")
                    && let syn::Expr::Lit(syn::ExprLit {
                        lit: Lit::Int(i), ..
                    }) = &nv.value
                {
                    numa = i.base10_parse()?;
                }
            }
        }
        Ok(InitArgs {
            topology,
            safety,
            workers,
            capacity,
            stack,
            numa,
        })
    }
}

#[proc_macro_attribute]
pub fn dtact_init(args: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(args as InitArgs);
    let input = parse_macro_input!(item as ItemFn);

    let topology = &args.topology;
    let safety = &args.safety;
    let workers = args.workers;
    let capacity = args.capacity;
    let stack = args.stack;
    let numa = args.numa;

    let topology_ident = syn::Ident::new(topology, input.sig.ident.span());
    let safety_ident = syn::Ident::new(safety, input.sig.ident.span());
    let autostart_fn_name = syn::Ident::new("dtact_autostart", input.sig.ident.span());

    let attrs = &input.attrs;
    let vis = &input.vis;
    let sig = &input.sig;
    let block = &input.block;

    let expanded = quote! {
        #[unsafe(no_mangle)]
        extern "C" fn #autostart_fn_name() {
            let runtime = dtact::GLOBAL_RUNTIME.get_or_init(|| {
                let mut workers_count = #workers;
                if workers_count == 0 {
                    workers_count = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
                }

                let scheduler = dtact::dta_scheduler::DtaScheduler::new(
                    workers_count,
                    dtact::dta_scheduler::TopologyMode::#topology_ident
                );

                let pool = dtact::memory_management::ContextPool::new(
                    #capacity,
                    #stack,
                    dtact::memory_management::SafetyLevel::#safety_ident,
                    #numa,
                    workers_count
                ).expect("DTA-V3 Hardware Initialization Failed");

                dtact::Runtime {
                    scheduler,
                    pool,
                    started: core::sync::atomic::AtomicBool::new(false),
                    shutdown: core::sync::atomic::AtomicBool::new(false),
                }
            });
            runtime.start();
        }

        #(#attrs)* #vis #sig {
            #autostart_fn_name();
            #block
        }
    };

    TokenStream::from(expanded)
}

/// Parsed `name = value` attribute args shared by [`dtact_io_init`],
/// [`fs_init`], and [`process_init`] — all three accept the same five
/// knobs (`workers`, `buffer_pool_size`, `chunk_size`, `ring_depth`,
/// `pin_cpus`) and each emits a call to its module's raw five-knob init
/// function in that module's canonical `(workers, ring_depth,
/// buffer_pool_size, chunk_size, pin_cpus)` argument order.
struct IoInitArgs {
    workers: usize,
    buffer_pool_size: usize,
    chunk_size: usize,
    pin_cpus: Vec<usize>,
    ring_depth: u32,
}

impl Parse for IoInitArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let vars = Punctuated::<Meta, Token![,]>::parse_terminated(input)?;
        let mut workers = 1;
        let mut buffer_pool_size = 65536;
        let mut chunk_size = 4096;
        let mut pin_cpus = Vec::new();
        let mut ring_depth = 256;

        for var in vars {
            if let Meta::NameValue(nv) = var {
                if nv.path.is_ident("workers") {
                    if let syn::Expr::Lit(syn::ExprLit {
                        lit: Lit::Int(i), ..
                    }) = &nv.value
                    {
                        workers = i.base10_parse()?;
                    }
                } else if nv.path.is_ident("buffer_pool_size") {
                    if let syn::Expr::Lit(syn::ExprLit {
                        lit: Lit::Int(i), ..
                    }) = &nv.value
                    {
                        buffer_pool_size = i.base10_parse()?;
                    }
                } else if nv.path.is_ident("chunk_size") {
                    if let syn::Expr::Lit(syn::ExprLit {
                        lit: Lit::Int(i), ..
                    }) = &nv.value
                    {
                        chunk_size = i.base10_parse()?;
                    }
                } else if nv.path.is_ident("ring_depth") {
                    if let syn::Expr::Lit(syn::ExprLit {
                        lit: Lit::Int(i), ..
                    }) = &nv.value
                    {
                        ring_depth = i.base10_parse()?;
                    }
                } else if nv.path.is_ident("pin_cpus")
                    && let syn::Expr::Array(syn::ExprArray { elems, .. }) = &nv.value
                {
                    for elem in elems {
                        if let syn::Expr::Lit(syn::ExprLit {
                            lit: Lit::Int(i), ..
                        }) = elem
                        {
                            pin_cpus.push(i.base10_parse()?);
                        }
                    }
                }
            }
        }
        Ok(IoInitArgs {
            workers,
            buffer_pool_size,
            chunk_size,
            pin_cpus,
            ring_depth,
        })
    }
}

/// Configures and starts whichever native `io` backend is active
/// (io_uring on Linux, IOCP on Windows, kqueue/mio elsewhere, or the
/// `tokio` wrapper — see `dtact_util::io::init_runtime`) before the
/// wrapped `main`/entry point runs. Takes the same five knobs as
/// [`fs_init`]/[`process_init`] (`workers`, `buffer_pool_size`,
/// `chunk_size`, `pin_cpus`, `ring_depth`), the "power user" counterpart
/// to the plain `dtact_util::io::init(workers)` convenience function.
#[proc_macro_attribute]
pub fn dtact_io_init(args: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(args as IoInitArgs);
    let input = parse_macro_input!(item as ItemFn);

    let workers = args.workers;
    let buffer_pool_size = args.buffer_pool_size;
    let chunk_size = args.chunk_size;
    let ring_depth = args.ring_depth;
    let pin_cpus = &args.pin_cpus;

    let attrs = &input.attrs;
    let vis = &input.vis;
    let sig = &input.sig;
    let block = &input.block;

    let expanded = quote! {
        #(#attrs)* #vis #sig {
            dtact_util::init_runtime(
                #workers,
                #ring_depth,
                #buffer_pool_size,
                #chunk_size,
                &[#(#pin_cpus),*],
            );
            #block
        }
    };

    TokenStream::from(expanded)
}

/// Sibling to [`dtact_io_init`] for the `fs` module: configures and starts
/// whichever native fs backend is active (io_uring slot pool on Linux,
/// IOCP slot pool on Windows, thread pool elsewhere — see
/// `dtact_util::fs::init_fs`) before the wrapped `main`/entry point runs.
/// Takes the same five knobs as `dtact_io_init` (`workers`,
/// `buffer_pool_size`, `chunk_size`, `pin_cpus`, `ring_depth`); `ring_depth`
/// is what actually sizes the preallocated op-slot pool for fs ops.
#[proc_macro_attribute]
pub fn fs_init(args: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(args as IoInitArgs);
    let input = parse_macro_input!(item as ItemFn);

    let workers = args.workers;
    let buffer_pool_size = args.buffer_pool_size;
    let chunk_size = args.chunk_size;
    let ring_depth = args.ring_depth;
    let pin_cpus = &args.pin_cpus;

    let attrs = &input.attrs;
    let vis = &input.vis;
    let sig = &input.sig;
    let block = &input.block;

    let expanded = quote! {
        #(#attrs)* #vis #sig {
            dtact_util::fs::init_fs(
                #workers,
                #ring_depth,
                #buffer_pool_size,
                #chunk_size,
                &[#(#pin_cpus),*],
            );
            #block
        }
    };

    TokenStream::from(expanded)
}

/// Sibling to [`fs_init`] for the `process` module: starts the
/// dtact-process blocking-thread pool before the wrapped `main`/entry
/// point runs. Same five knobs for call-site parity with `fs_init`/
/// `dtact_io_init`; only `workers` is actually consulted by
/// `dtact_util::process::init_process` today (see that function's doc
/// for why the ring/buffer-pool knobs don't apply to a thread-pool-
/// bridged backend).
#[proc_macro_attribute]
pub fn process_init(args: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(args as IoInitArgs);
    let input = parse_macro_input!(item as ItemFn);

    let workers = args.workers;
    let buffer_pool_size = args.buffer_pool_size;
    let chunk_size = args.chunk_size;
    let ring_depth = args.ring_depth;
    let pin_cpus = &args.pin_cpus;

    let attrs = &input.attrs;
    let vis = &input.vis;
    let sig = &input.sig;
    let block = &input.block;

    let expanded = quote! {
        #(#attrs)* #vis #sig {
            dtact_util::process::init_process(
                #workers,
                #ring_depth,
                #buffer_pool_size,
                #chunk_size,
                &[#(#pin_cpus),*],
            );
            #block
        }
    };

    TokenStream::from(expanded)
}
