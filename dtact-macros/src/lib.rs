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

/// Resolves a `stack = "256K"` style attribute value to a byte count at
/// compile time, so the generated code carries a plain integer and spawning
/// the task never parses a string.
///
/// Mirrors `dtact::api::fiber::parse_stack_size` (the runtime parser used by
/// `spawn_with_stack`): an optional binary suffix (`K`/`KiB`, `M`/`MiB`,
/// `G`/`GiB`) applied to a decimal number. Returns `None` for anything it
/// cannot resolve, which the expansion turns into "no explicit request" —
/// the fiber then gets the slot's full stack instead of failing to compile.
fn stack_size_bytes(spec: &str) -> Option<usize> {
    let spec = spec.trim();
    let split = spec
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(spec.len());
    let (digits, suffix) = spec.split_at(split);
    let value: usize = digits.parse().ok()?;
    if value == 0 {
        return None;
    }
    let unit = match suffix.trim().to_ascii_uppercase().as_str() {
        "" | "B" => 1usize,
        "K" | "KB" | "KIB" => 1024,
        "M" | "MB" | "MIB" => 1024 * 1024,
        "G" | "GB" | "GIB" => 1024 * 1024 * 1024,
        _ => return None,
    };
    value.checked_mul(unit)
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

    // `stack = "..."` resolved to bytes here, at compile time.
    let stack_bytes = stack_size_bytes(stack);
    let stack_expr = match stack_bytes {
        Some(bytes) => quote! { ::core::option::Option::Some(#bytes) },
        None => quote! { ::core::option::Option::None },
    };
    let stack_bytes_const = match stack_bytes {
        Some(bytes) => quote! { ::core::option::Option::Some(#bytes) },
        None => quote! { ::core::option::Option::None },
    };

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
                stack_size: #stack_expr,
                _marker: std::marker::PhantomData,
            }
        }

        pub mod #metadata_mod {
            pub const PRIORITY: dtact::Priority = dtact::Priority::#priority_ident;
            pub const AFFINITY: dtact::topology::Affinity = dtact::topology::Affinity::#affinity_ident;
            pub const KIND: dtact::WorkloadKind = dtact::WorkloadKind::#kind_ident;
            pub const STACK_SIZE: &'static str = #stack;
            /// `STACK_SIZE` resolved to bytes at compile time, or `None` if
            /// the attribute value could not be parsed.
            pub const STACK_SIZE_BYTES: ::core::option::Option<usize> = #stack_bytes_const;
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

/// Parsed arguments of [`export_fiber`].
///
/// Every field is optional and defaults to the runtime's own default, so a
/// bare `#[export_fiber]` behaves exactly as before these knobs existed.
/// Before this existed the attribute accepted an argument list and silently
/// discarded it, so `#[export_fiber(stack = "8M")]` produced a fiber with
/// the default stack and no way to tell.
#[derive(Default)]
struct ExportFiberArgs {
    stack: Option<usize>,
    priority: Option<String>,
    safety: Option<String>,
    name: Option<String>,
}

impl Parse for ExportFiberArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let vars = Punctuated::<Meta, Token![,]>::parse_terminated(input)?;
        let mut args = Self::default();
        for var in vars {
            let Meta::NameValue(nv) = var else {
                return Err(syn::Error::new_spanned(
                    var,
                    "expected `name = value` (stack, priority, safety, name)",
                ));
            };
            let str_value = || -> syn::Result<String> {
                match &nv.value {
                    syn::Expr::Lit(syn::ExprLit {
                        lit: Lit::Str(s), ..
                    }) => Ok(s.value()),
                    other => Err(syn::Error::new_spanned(
                        other,
                        "expected a string literal, e.g. `stack = \"256K\"`",
                    )),
                }
            };
            if nv.path.is_ident("stack") {
                let spec = str_value()?;
                args.stack = Some(stack_size_bytes(&spec).ok_or_else(|| {
                    syn::Error::new_spanned(
                        &nv.value,
                        "cannot parse stack size; expected \"256K\", \"2M\", ...",
                    )
                })?);
            } else if nv.path.is_ident("priority") {
                args.priority = Some(str_value()?);
            } else if nv.path.is_ident("safety") {
                args.safety = Some(str_value()?);
            } else if nv.path.is_ident("name") {
                args.name = Some(str_value()?);
            } else {
                return Err(syn::Error::new_spanned(
                    &nv.path,
                    "unknown export_fiber argument (expected stack, priority, safety, name)",
                ));
            }
        }
        Ok(args)
    }
}

/// Exports a plain (blocking) `fn` as a C-callable fiber launcher.
///
/// Accepts the same spawn-time knobs as the Rust API, resolved at compile
/// time:
/// * `stack = "256K"` — stack size for the exported fiber
/// * `priority = "High"` — `Low`/`Normal`/`High`/`Critical`
/// * `safety = "Safety1"` — per-fiber stack-overflow policy
/// * `name = "db-writer"` — name used in diagnostics and `fiber_info`
///
/// Omitted knobs fall back to the runtime defaults.
#[proc_macro_attribute]
pub fn export_fiber(args: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(args as ExportFiberArgs);
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

    let mut opts = quote! { dtact::api::fiber::ClosureSpawnOptions::new() };
    if let Some(bytes) = args.stack {
        opts = quote! { #opts.stack_size(#bytes) };
    }
    if let Some(priority) = &args.priority {
        let ident = syn::Ident::new(priority, fn_name.span());
        opts = quote! { #opts.priority(dtact::Priority::#ident) };
    }
    if let Some(safety) = &args.safety {
        let ident = syn::Ident::new(safety, fn_name.span());
        opts = quote! { #opts.safety(dtact::memory_management::SafetyLevel::#ident) };
    }
    if let Some(name) = &args.name {
        opts = quote! { #opts.name(#name) };
    }

    let expanded = quote! {
        #input

        #[unsafe(no_mangle)]
        pub extern "C" fn #wrapper_name(#(#c_params),*) -> dtact::dtact_handle_t {
            dtact::api::fiber::spawn_closure(
                move || {
                    #fn_name(#(#call_args),*);
                },
                #opts,
            )
        }
    };

    TokenStream::from(expanded)
}

/// Reads a `name = 123` or `name = "123"` attribute value as an integer.
///
/// Both spellings are accepted because the crate's own documentation and
/// examples use both; before this, a quoted value was silently discarded and
/// the argument kept its default — the setting appeared to be honoured while
/// having no effect at all. A non-numeric value is a hard error rather than a
/// silent fallback, for the same reason.
fn int_arg(expr: &syn::Expr) -> syn::Result<Option<usize>> {
    match expr {
        syn::Expr::Lit(syn::ExprLit {
            lit: Lit::Int(i), ..
        }) => Ok(Some(i.base10_parse()?)),
        syn::Expr::Lit(syn::ExprLit {
            lit: Lit::Str(s), ..
        }) => s.value().trim().parse::<usize>().map(Some).map_err(|_| {
            syn::Error::new_spanned(expr, "expected an integer or a string of digits")
        }),
        _ => Err(syn::Error::new_spanned(
            expr,
            "expected an integer literal or a string literal",
        )),
    }
}

/// Reads a `stack = 262144`, `stack = "262144"` or `stack = "256K"`
/// attribute value as a byte count.
fn stack_arg(expr: &syn::Expr) -> syn::Result<Option<usize>> {
    if let syn::Expr::Lit(syn::ExprLit {
        lit: Lit::Str(s), ..
    }) = expr
    {
        let value = s.value();
        return stack_size_bytes(&value).map(Some).ok_or_else(|| {
            syn::Error::new_spanned(
                expr,
                "cannot parse stack size; expected bytes (\"262144\") or a \
                 binary suffix (\"256K\", \"2M\", \"1G\")",
            )
        });
    }
    int_arg(expr)
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
                    if let Some(value) = int_arg(&nv.value)? {
                        workers = value;
                    }
                } else if nv.path.is_ident("capacity") {
                    if let Some(value) = int_arg(&nv.value)? {
                        capacity = u32::try_from(value).map_err(|_| {
                            syn::Error::new_spanned(&nv.value, "`capacity` does not fit in a u32")
                        })?;
                    }
                } else if nv.path.is_ident("stack") {
                    if let Some(value) = stack_arg(&nv.value)? {
                        stack = value;
                    }
                } else if nv.path.is_ident("numa")
                    && let Some(value) = int_arg(&nv.value)?
                {
                    numa = value;
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
